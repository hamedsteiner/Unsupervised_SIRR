'''
Date: March, 2021 
Author: Hamed RahmaniKhezri 

This file is originally from "'Double DIP" (https://github.com/yossigandelsman/DoubleDIP) 
Some modifications are built to define the baselines
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
from net.spectral import SpectralNorm
from net.sagan_models import Self_Attn
from net.resnet_model import resnet18 as Resnet18
from utils.image_io import *


def init_weights(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 1.0)
        #nn.init.xavier_uniform_(m.weight, 0.5) 
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('Norm2d') != -1: # batchnorm or instancenorm 
        m.weight.data.normal_(1.0, 0.01)
        nn.weight.data.fill_(0.5) 
        


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            #nn.Conv2d(in_ch, out_ch, 3, padding=1),
            SpectralNorm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
            #nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(),
            #nn.Conv2d(out_ch, out_ch, 3, padding=1),
            SpectralNorm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
            #  nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            #nn.AvgPool2d(2),
            #nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=2), 
            SpectralNorm(nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=2)), 
            nn.LeakyReLU(),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up_concat(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up_concat, self).__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'), 
                nn.ReflectionPad2d((1, 1, 1, 1)), 
                nn.Conv2d(in_ch//2, in_ch//2, (3, 3))
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
            )

        self.conv =  nn.Sequential(
                nn.LeakyReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(in_ch, out_ch, (3, 3)),
                nn.LeakyReLU())
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:# resize-conv
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bicubic'), 
                nn.ReflectionPad2d((1, 1, 1, 1)), 
                nn.Conv2d(in_ch, out_ch, (3, 3))
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            )

        self.conv =  nn.Sequential(
                nn.LeakyReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)), 
                nn.Conv2d(out_ch, out_ch, (3, 3)),
                nn.LeakyReLU())
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class up_spec(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up_spec, self).__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'), 
                nn.ReflectionPad2d((1, 1, 1, 1)), 
                SpectralNorm(nn.Conv2d(in_ch, out_ch, (3, 3)))
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            )

        self.conv =  nn.Sequential(
                nn.LeakyReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)), 
                SpectralNorm(nn.Conv2d(out_ch, out_ch, (3, 3))),
                nn.LeakyReLU())
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, connection=[True]*4, enable_sigmoid=False):
        super(UNet, self).__init__()
        self.connection = connection 

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up_concat(1024, 256) if self.connection[0] else up(512, 256)
        self.up2 = up_concat(512, 128)  if self.connection[1] else up(256, 128)
        self.up3 = up_concat(256, 64)   if self.connection[2] else up(128, 64)
        self.up4 = up_concat(128, 64)   if self.connection[3] else up(64, 64)
        self.outc = outconv(64, n_classes)

        self.enable_sigmoid = enable_sigmoid 
        self.sigmoid = nn.Sigmoid()

        # xavier initialization 
        init_weights(self)
        
    def enc(self, x): 
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]

    def dec(self, x_list): 
        x1, x2, x3, x4, x5 = x_list
        x6 = self.up1(x5, x4) if self.connection[0] else self.up1(x5)
        x7 = self.up2(x6, x3) if self.connection[1] else self.up2(x6)
        x8 = self.up3(x7, x2) if self.connection[2] else self.up3(x7)
        x9 = self.up4(x8, x1) if self.connection[3] else self.up4(x8)
        y = self.outc(x9)
        y = self.sigmoid(y) if self.enable_sigmoid else y 
        return y, [x5, x6, x7, x8, x9]

    def forward(self, x):
        self.enc_out = self.enc(x)
        y, self.dec_out = self.dec(self.enc_out)

        # for latent space separation
        self.latent = self.enc_out[-1]
        return y


class UNetRes(nn.Module): 
    def __init__(self, n_channels, n_classes, connection=[True]*4, pretrained=None, enable_sigmoid=False): 
        super(UNetRes, self).__init__()
        self.connection = connection 

        if pretrained == 'places365':
            # load the pre-trained weights           
            pretrained_file = './models/resnet18_places365.pth.tar' 
            checkpoint = torch.load(pretrained_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            resnet18 = models.__dict__['resnet18'](num_classes=365) 
            resnet18.load_state_dict(state_dict)
        elif pretrained == 'imagenet': 
            pretrained_file = './models/resnet18-5c106cde.pth'
            resnet18 = models.resnet18()
            resnet18.load_state_dict(torch.load(pretrained_file))
        else: 
            resnet18 = models.resnet18()

        enc_layers = list(resnet18.children()) # [1, 3, 224, 224]
        if n_channels != 3: 
            enc_layers[0] = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.down1 = nn.Sequential(*enc_layers[:5])   #[1,64,56,56] 
        self.down2 = nn.Sequential(*enc_layers[5])   #[1,128,28,28]
        self.down3 = nn.Sequential(*enc_layers[6])   #[1,256,14,14] 
        self.down4 = nn.Sequential(*enc_layers[7])   #[1,512,7,7] 
        self.bottleneck = double_conv(512,256)
        self.up1 = up_concat(512, 128) if connection[0] else up(256, 128)
        self.up2 = up_concat(256, 64)  if connection[1] else up(128, 64)
        self.up3 = up_concat(128, 64)  if connection[2] else up(64, 64)
        self.up4 = up(64, 64) 
        self.up5 = up(64, 64)
        self.outc = outconv(64, n_classes)  

        self.enable_sigmoid = enable_sigmoid 
        self.sigmoid = nn.Sigmoid()

        self.latent = None 

        # xavier initialization wihout enc 
        init_weights(self.bottleneck)
        init_weights(self.up1)
        init_weights(self.up2)
        init_weights(self.up3)
        init_weights(self.up4)
        init_weights(self.up5)
        init_weights(self.outc)
        
    def enc(self, x): 
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.bottleneck(x4)
        return [x1, x2, x3, x4, x5]

    def dec(self, x_list): 
        x1, x2, x3, x4, x5 = x_list
        x6 = self.up1(x5, x3) if self.connection[0] else self.up1(x5) # 7->14
        x7 = self.up2(x6, x2) if self.connection[1] else self.up2(x6) # 14->28
        x8 = self.up3(x7, x1) if self.connection[2] else self.up3(x7) # 28->56
        x9 = self.up5(self.up4(x8)) # 56->224
        y = self.outc(x9)
        y = self.sigmoid(y) if self.enable_sigmoid else y 
        return y, [x6, x7, x8, x9]

    def forward(self, x):
        self.enc_out = self.enc(x)
        y, self.dec_out = self.dec(self.enc_out)

        # for latent space separation
        self.latent = self.enc_out[-1]
        return y

class FeatureExtractor(nn.Module): 
    def __init__(self, model_name, size=(1,1,224,224)): 
        super(FeatureExtractor, self).__init__()
        if model_name == 'imagenet': 
            pretrained_file = './models/resnet18-5c106cde.pth'
            perceptual_model = models.resnet18()
            perceptual_model.load_state_dict(torch.load(pretrained_file))
        else: #'places365'
            pretrained_file = './models/resnet18_places365.pth.tar' 
            checkpoint = torch.load(pretrained_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            perceptual_model = models.__dict__['resnet18'](num_classes=365)
            perceptual_model.load_state_dict(state_dict)

        perceptual_model.eval()

        enc_layers = list(perceptual_model.children())
        self.layer_list = [] 
        # self.layer1 = nn.Sequential(*enc_layers[:5])   #[1,64,56,56] 
        # self.layer2 = nn.Sequential(*enc_layers[5])   #[1,128,28,28]
        # self.layer3 = nn.Sequential(*enc_layers[6])   #[1,256,14,14] 
        # self.layer4 = nn.Sequential(*enc_layers[7])   #[1,512,7,7] 
        self.layer_list.append(nn.Sequential(*enc_layers[:6])) # [1, 128,28,28]
        self.layer_list.append(nn.Sequential(*enc_layers[6:8])) # [1,512,7,7]

        self.upsample = nn.Upsample(size=size[-2:], mode='bilinear', align_corners=True)

    def forward(self, img): 
        x = img
        feat_list = []
        for i in range(len(self.layer_list)): 
            y = self.layer_list[i].to(str(img.device))(x)
            feat_list.append(self.upsample(y))    
            x = y 
        return torch.cat(feat_list, dim=1)



class PercepNet_(nn.Module): 
    def __init__(self, n_channels, n_classes,  pretrained='places365'): 
        super(PercepNet_, self).__init__()
        
        # Extract features from input mixed image 
        if pretrained == 'places365':
            # load the pre-trained weights           
            pretrained_file = './models/resnet18_places365.pth.tar' 
            checkpoint = torch.load(pretrained_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            pretrained_model = Resnet18(num_classes=365) 
            pretrained_model.load_state_dict(state_dict)
        elif pretrained == 'imagenet': 
            pretrained_file = './models/resnet18-5c106cde.pth'
            state_dict = torch.load(pretrained_file)
            pretrained_model = Resnet18(num_classes=1000)
            pretrained_model.load_state_dict(state_dict)
        else: 
            resnet18 = models.resnet18()
        self.feature_extractor = pretrained_model 

        # fix the gradient 
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.down1 = down(128, 64) # input : feature14(128,28,28)
        self.down2 = down(256+64, 128) # input : feature7(256, 14, 14) + down1
        self.down3 = double_conv(512+128, 256) # input : feature4(512, 7,7) 

        self.up1 = up(256, 128)
        self.up2 = up(128, 64)
        self.up3 = up(64, 64)
        self.up4 = up(64, 64) 
        self.up5 = up(64, 64)
        self.outc = outconv(64, n_classes)  

        self.sigmoid = nn.Sigmoid()

        self.latent = None 

        # xavier initialization
        init_weights(self.down1)
        init_weights(self.down2)
        init_weights(self.down3)
        init_weights(self.up1)
        init_weights(self.up2)
        init_weights(self.up3)
        init_weights(self.up4)
        init_weights(self.up5)
        init_weights(self.outc)

    def enc(self, feats):
        output14 = self.down1(feats[1].detach())
        pred14 = torch.cat((output14, feats[2].detach()), dim=1)
        output7 = self.down2(pred14)
        pred7 = torch.cat((output7, feats[3].detach()), dim=1)
        
        pred4 = self.down3(pred7)
        return pred4

    def dec(self, x):       
        x = self.up1(x) # 7->14
        x = self.up2(x) # 14->28
        x = self.up3(x) # 28->56
        x = self.up5(self.up4(x)) # 56->224
        y = self.outc(x)
        return y 

    def forward(self, x):
        _, feats = self.feature_extractor(x) 
        self.latent =  self.enc(feats)
        y  = self.dec(self.latent) 
        return y

class AlphaNet(nn.Module): 
    def __init__(self, n_channels, n_classes, image_t, pretrained='places365'): 
        super(AlphaNet, self).__init__()
        
        # Extract features from input mixed image 
        if pretrained == 'places365':
            # load the pre-trained weights           
            pretrained_file = './models/resnet18_places365.pth.tar' 
            checkpoint = torch.load(pretrained_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            pretrained_model = Resnet18(num_classes=365) 
            pretrained_model.load_state_dict(state_dict)
        elif pretrained == 'imagenet': 
            pretrained_file = './models/resnet18-5c106cde.pth'
            state_dict = torch.load(pretrained_file)
            pretrained_model = Resnet18(num_classes=1000)
            pretrained_model.load_state_dict(state_dict)
        else: 
            pretrained_model = models.resnet18()
        self.feature_extractor = pretrained_model 

        # fix the gradient 
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.feature_extractor.to(str(image_t.device))
        
        # extract the feat 
        _, self.feats = self.feature_extractor(renormalize(image_t)) 
        
        self.feat1 = nn.Conv2d(128, 16, 1) # input: feature14(128, 28,28)
        self.down1 = down(16, 16)
        self.feat2 = nn.Conv2d(256, 32, 1) # input: feature7(256, 14, 14) 
        self.down2 = down(32+16, 32)
        self.feat3 = nn.Conv2d(512, 64, 1) # input : feature4(512, 7,7) 
        self.down3 = down(64+32, 64)
        self.down4 = down(64, 32)
        self.down5 = down(32, 16) # 1X16X1x1
        self.softmax = nn.Softmax(dim=1) 
        self.linear = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()
        
        # xavier initialization
        init_weights(self.down1)
        init_weights(self.down2)
        init_weights(self.down3)
        init_weights(self.down4)
        init_weights(self.down5)
        init_weights(self.feat1)
        init_weights(self.feat2)
        init_weights(self.feat3)

    def forward(self, x):
        feat1 = self.feat1(self.feats[1].detach())
        out1  = self.down1(feat1)
        feat2 = self.feat2(self.feats[2].detach())
        out2  = self.down2(torch.cat((out1, feat2), dim=1))
        feat3 = self.feat3(self.feats[3].detach())
        out3  = self.down3(torch.cat((out2, feat3), dim=1))
        out4  = self.down4(out3)
        out5  = self.down5(out4)
        #out   = self.sigmoid(self.linear(torch.flatten(out5)))
       # return out.unsqueeze(-1) 
        out   = self.softmax(out5)
        return torch.max(out.view(-1))


class PercepNet(nn.Module): 
    def __init__(self, n_channels, n_classes, image_t,  pretrained='places365',enable_attn=False): 
        super(PercepNet, self).__init__()
        
        self.enable_attn = enable_attn

        # Extract features from input mixed image 
        if pretrained == 'places365':
            # load the pre-trained weights           
            pretrained_file = './models/resnet18_places365.pth.tar' 
            checkpoint = torch.load(pretrained_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            pretrained_model = Resnet18(num_classes=365) 
            pretrained_model.load_state_dict(state_dict)
        elif pretrained == 'imagenet': 
            pretrained_file = './models/resnet18-5c106cde.pth'
            state_dict = torch.load(pretrained_file)
            pretrained_model = Resnet18(num_classes=1000)
            pretrained_model.load_state_dict(state_dict)
        else: 
            resnet18 = models.resnet18()
        self.feature_extractor = pretrained_model 

        # fix the gradient 
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.feature_extractor.to(str(image_t.device))

        # extract the feat 
        _, self.feats = self.feature_extractor(renormalize(image_t)) 

        self.input = nn.Sequential(
                    down(n_channels, 16), # 224 -> 112 
                    down(16, 32),  # 122 -> 56
                    down(32, 32),  # 56 -> 28
        )
        #self.down0 = down(64+16, 32) # input : feature56(64, 56, 56)=> increase detecting patterns more than shape
        self.down1 = down(128+32, 64) # input : feature14(128,28,28)
        self.down2 = down(256+64, 128) # input : feature7(256, 14, 14) + down1
        self.down3 = double_conv(512+128, 128) # input : feature4(512, 7,7) 

        self.decoder = nn.Sequential(
                    up(128, 128),
                    up(128, 64),
                    up(64, 64),
                    up(64, 64),
                    up(64, 64),
        )
        self.output = outconv(64, n_classes)
    
        self.latent = None 

        if self.enable_attn: 
            self.attn = Self_Attn(128, 'relu')

        # xavier initialization
        init_weights(self.input)
        init_weights(self.down1)
        init_weights(self.down2)
        init_weights(self.down3)
        init_weights(self.decoder)
        init_weights(self.output)
        
        # check parameter numbers 
        inp=dw=dec=outp=1
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.find('decoder') > -1 : 
                    dec +=np.prod(list(param.size())) 
                elif name.find('input') > -1: 
                    inp +=np.prod(list(param.size())) 
                elif name.find('down') > -1: 
                    dw +=np.prod(list(param.size())) 
                elif name.find('output') > -1: 
                    dw +=np.prod(list(param.size())) 

        print("inp:{:,d}, down:{:,d}, dec:{:,d}, out:{:d}".format(inp, dw, dec, outp))

    def enc(self, x):
        #in0 = torch.cat([self.input(x),   self.feats[0].detach()], dim=1)
        in1 = torch.cat([self.input(x),   self.feats[1].detach()], dim=1)
        in2 = torch.cat([self.down1(in1), self.feats[2].detach()], dim=1)
        in3 = torch.cat([self.down2(in2), self.feats[3].detach()], dim=1)
        y = self.down3(in3)
        if self.enable_attn: 
            y, attn_map = self.attn(y)
        return y 

    def dec(self, x): 
        y = self.decoder(x)
        y = self.output(y) 
        return y

    def forward(self, x):
        self.latent = self.enc(x)
        y  = self.dec(self.latent) 
        return y


if __name__ == '__main__': 
    import sys 
    sys.path.append('../WildDIP')
    from utils.image_io import *

    img_pil = crop_image(get_image('images/m.jpg', 224)[0], d=32)
    image = pil_to_np(img_pil)
    image_torch = np_to_torch(image)

    # unet = UNet(3,3) 
    # unet(image_torch)
    
