'''
Dialated in encoder
Date: March, 2021 
Author: Hamed RahmaniKhezri 
'''














"""
Skip connection 
with Res-net 

Exp3 -> 1
28 and 112 , but only middle skip

"""





import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
from net.spectral import SpectralNorm
from net.sagan_models import Self_Attn
from net.resnet_model import resnet18 as Resnet18
from net.resnet_model import resnet50 as Resnet50

from utils.image_io import *

import numpy as np

def init_weights(m):
    torch.cuda.manual_seed_all(1943)
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 1.0)
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
            SpectralNorm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
            nn.LeakyReLU(),
            SpectralNorm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class test1(nn.Module): # H3 to try f=down3 as dilated
    def __init__(self, in_ch, out_ch):
        super(test1, self).__init__()
        self.conv = nn.Sequential(
        SpectralNorm(nn.Conv2d(in_ch, out_ch, 2, padding=1, dilation = 2)),
        nn.LeakyReLU()            
        )
    def forward(self, x):
        x = self.conv(x)
        return x 



        

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=2)), 
            nn.LeakyReLU(),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
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

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class PercepNet(nn.Module): 
    def __init__(self, n_channels, n_classes, image_t,  pretrained='places365',enable_attn=False): 
        super(PercepNet, self).__init__()
        
        torch.cuda.manual_seed_all(1943)
        self.enable_attn = enable_attn

        # Extract features from input mixed image 
        if pretrained == 'places365':
            # load the pre-trained weights           
            pretrained_file = './models/resnet18_places365.pth.tar' 
            checkpoint = torch.load(pretrained_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            pretrained_model = Resnet18(num_classes=365) 
            pretrained_model.load_state_dict(state_dict)
            print("percepnet = plces365")
        elif pretrained == 'imagenet': 
            pretrained_file = './models/resnet18-5c106cde.pth'
            state_dict = torch.load(pretrained_file)
            pretrained_model = Resnet18(num_classes=1000)
            pretrained_model.load_state_dict(state_dict)
            print("percepnet = imagenet")
        else: 
            pretrained_model = Resnet18(num_classes=100)
            print("percepnet = none")
        self.feat_extractor = pretrained_model 

        # fix the gradient 
        for param in self.feat_extractor.parameters():
            param.requires_grad = False
        self.feat_extractor.eval()
        self.feat_extractor.to(str(image_t.device))

        # extract the feat 
        _, self.feats = self.feat_extractor(renormalize(image_t)) 



        self.input = nn.Sequential(
                    down(n_channels, 16), # 224 -> 112 
                    down(16, 32),  # 112 -> 56
                    down(32, 32),  # 56 -> 28
        )
        #self.down0 = down(64+16, 32) # input : feature56(64, 56, 56)=> increase detecting patterns more than shape


        self.down1 = down(128+32, 64) # input : feature14(128,28,28)
        self.down2 = down(256+64, 128) # input : feature7(256, 14, 14) + down1


        """
        TO DO : not use Down2 and use one less Conv?
        """

        self.inp_down1 = down(n_channels, 16)
        self.inp_down2 = down(16, 32)
        self.inp_down3 = down(32, 32)





        # self.down3 = double_conv(512+128, 128) # input : feature4(512, 7,7)
        """
        TO DO : new Down 3 based on test 1 
        """
        self.down3 = test1(512+128, 128)



        self.decoder = nn.Sequential(
                    up(128, 128),
                    up(128, 64),
                    up(64, 64),
                    up(64, 64),
                    up(64, 64),
        )
        self.output = outconv(64, n_classes)
        if self.enable_attn: 
            self.attn = Self_Attn(128, 'relu')

        self.Up1 =  up(128, 128)
        self.Up2 =  up(128, 64)
        self.Up3 =  up(80, 64)
        self.Up4 =  up(64, 64)
        self.Up5 =  up(72, 64)
            
                            

        # xavier initialization
        init_weights(self.inp_down1)
        init_weights(self.inp_down2)
        init_weights(self.inp_down3)

        init_weights(self.down1)
        init_weights(self.down2)
        init_weights(self.down3)


        # init_weights(self.decoder)

        init_weights(self.Up1)
        init_weights(self.Up2)
        init_weights(self.Up3)
        init_weights(self.Up4)
        init_weights(self.Up5)

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




    # def enc(self, x):
    #     #in0 = torch.cat([self.input(x),   self.feats[0].detach()], dim=1)



        

    #     return y 

    # def dec(self, x): 
    #     y = self.decoder(x)
    #     y = self.output(y) 

    #     # print("self.decoder(x) ", self.decoder(x).shape )
    #     # print("self.output(y)  ",y.shape )
    #     # exit()
    #     return y

    def forward(self, x):
        # print("self.enc(x)  ",self.enc(x).shape )



        in_down1 = self.inp_down1(x) #112
        in_down2 = self.inp_down2(in_down1) #56
        in_down3 = self.inp_down3(in_down2) #28
        
        in1 = torch.cat([in_down3,   self.feats[1].detach()], dim=1) # 14
        
        # print("Original x ", x.shape )
        # print("self.input(x) ", self.input(x).shape )
        # print("self.feats[1].detach() ", self.feats[1].detach().shape )
        # print("In1 ",in1.shape)
        
        in2 = torch.cat([self.down1(in1), self.feats[2].detach()], dim=1) # 7 
        
        # print("self.down1(in1) ", self.down1(in1).shape )
        # print("self.feats[2].detach() ", self.feats[2].detach().shape )
        # print("In2 ",in2.shape)
        
        
        in3 = torch.cat([self.down2(in2), self.feats[3].detach()], dim=1) # 7 
        y_enc = self.down3(in3)
        if self.enable_attn: 
            y_enc, attn_map = self.attn(y_enc)
    
        # print("self.down2(in2) ", self.down2(in2).shape )
        # print("self.feats[3].detach() ", self.feats[3].detach().shape )
        # print("In3 ",in3.shape)
        # print("Y ",y.shape)
        # exit()


        up1 = self.Up1(y_enc) #14
        up2 = self.Up2(up1) #28

        # up3 = self.Up3(up2) #56



        

        up3 = self.Up3(torch.cat([up2, in_down3[:, 8:24, :, :]], dim=1)) #56

        up4 = self.Up4(up3) #112

        # up5 = self.Up5(up4) #224
        up5 = self.Up5(torch.cat([up4, in_down1[:, 4:12, :, :]], dim=1)) #224

        # print("Y ",in_down1[:, 4:12, :, :].shape)
        # exit()

        y = self.output(up5) 
        
        return y














class PercepNet0(nn.Module): 
    def __init__(self, n_channels, n_classes, image_t,  pretrained='places365',enable_attn=False): 
        super(PercepNet0, self).__init__()
        
        torch.cuda.manual_seed_all(1943)
        self.enable_attn = enable_attn

        # Extract features from input mixed image 
        if pretrained == 'places365':
            # load the pre-trained weights           
            pretrained_file = './models/resnet18_places365.pth.tar' 
            checkpoint = torch.load(pretrained_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            pretrained_model = Resnet18(num_classes=365) 
            pretrained_model.load_state_dict(state_dict)
            print("percepnet = plces365")
        elif pretrained == 'imagenet': 
            pretrained_file = './models/resnet18-5c106cde.pth'
            state_dict = torch.load(pretrained_file)
            pretrained_model = Resnet18(num_classes=1000)
            pretrained_model.load_state_dict(state_dict)
            print("percepnet = imagenet")
        else: 
            pretrained_model = Resnet18(num_classes=100)
            print("percepnet = none")
        self.feat_extractor = pretrained_model 

        # fix the gradient 
        for param in self.feat_extractor.parameters():
            param.requires_grad = False
        self.feat_extractor.eval()
        self.feat_extractor.to(str(image_t.device))

        # extract the feat 
        _, self.feats = self.feat_extractor(renormalize(image_t)) 

        self.input = nn.Sequential(
                    down(n_channels, 16), # 224 -> 112 
                    down(16, 32),  # 122 -> 56
                    down(32, 32),  # 56 -> 28
        )
        #self.down0 = down(64+16, 32) # input : feature56(64, 56, 56)=> increase detecting patterns more than shape
        self.down1 = down(32, 64) # input : feature14(128,28,28)
        self.down2 = down(64, 128) # input : feature7(256, 14, 14) + down1
        self.down3 = double_conv(128, 128) # input : feature4(512, 7,7) 

        self.decoder = nn.Sequential(
                    up(128, 128),
                    up(128, 64),
                    up(64, 64),
                    up(64, 64),
                    up(64, 64),
        )
        self.output = outconv(64, n_classes)
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
        in1 = self.input(x) 
        in2 = self.down1(in1) 
        in3 = self.down2(in2)
        y = self.down3(in3)
        if self.enable_attn: 
            y, attn_map = self.attn(y)
        return y 

    def dec(self, x): 
        y = self.decoder(x)
        y = self.output(y) 
        return y

    def forward(self, x):
        return self.dec(self.enc(x))


    
