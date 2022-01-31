

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .spectral import SpectralNorm
import numpy as np
import torchvision.models as models 

class Self_Attn(nn.Module):
    """ Self attention Layer
        all the implementations below are from https://github.com/heykeetae/Self-Attention-GAN
    """
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out, attention

class Generator(nn.Module):
    """Generator.
        all the implementations below are from https://github.com/heykeetae/Self-Attention-GAN
    """

    def __init__(self, batch_size=1, z_dim=1000, ch_out=3, image_size=64, conv_dim=32):
        super(Generator, self).__init__()
        self.n_layer = int(round(np.log2(image_size)))-2 # 64->4, 225->6
       
        curr_dim = conv_dim * (2 ** self.n_layer) 
        for l in range(self.n_layer):
            layer = []
            if l == 0: 
                k_size = 4 if image_size % (2 ** self. n_layer) == 0 else 3 
                layer.append(SpectralNorm(nn.ConvTranspose2d(z_dim, curr_dim//2, k_size)))
                #layer.append(nn.ConvTranspose2d(z_dim, curr_dim//2, k_size))

            else: 
                output_padding = 1 if k_size == 3 and l == 1 else 0 
                layer.append(SpectralNorm(nn.ConvTranspose2d(curr_dim,curr_dim//2, 4, 2, 1, output_padding)))
                #layer.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1,
                #output_padding))
            layer.append(nn.ReLU())
            setattr(self, 'l%d'%(l+1), nn.Sequential(*layer))
            curr_dim = curr_dim // 2
        
        self.last = nn.Sequential(nn.ConvTranspose2d(curr_dim, ch_out, 4, 2, 1),
                                  nn.Sigmoid()) # nn.Tanh()
       
        self.attn1 = Self_Attn(curr_dim * 2 , 'relu')
        self.attn2 = Self_Attn(curr_dim,  'relu')

    def forward(self, z):
        enable_attn = True
        x = z.view(z.size(0), z.size(1),1, 1)
        p1, p2 = None, None 
        for l in range(self.n_layer):
            if enable_attn and l==self.n_layer-1:
                x, p1 = self.attn1(x) 
            x = getattr(self, 'l%d'%(l+1))(x)
        if enable_attn: 
            x, p2 = self.attn2(x) 
        x=self.last(x)
        return x, p1, p2

class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier.
    all the implementations below are from https://github.com/heykeetae/Self-Attention-GAN
    """

    def __init__(self, batch_size=1, ch_in=3, z_dim=1000, image_size=64, conv_dim=32):
        super(Discriminator, self).__init__()
        self.n_layer = int(round(np.log2(image_size)))-2 # 64->4, 225->6
       
        curr_dim = conv_dim // 2
        for l in range(self.n_layer):
            layer = []
            if l == 0: 
                layer.append(SpectralNorm(nn.Conv2d(ch_in, curr_dim*2, 4, 2, 1)))
                #layer.append(nn.Conv2d(3, curr_dim*2, 4, 2, 1))
            else: 
                layer.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1)))
                #layer.append(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1))
            layer.append(nn.LeakyReLU(0.1))
            setattr(self, 'l%d'%(l+1), nn.Sequential(*layer))
            curr_dim = curr_dim * 2
        
        k_size = 4 if image_size % (2 ** self. n_layer) == 0 else 3 
        self.last = nn.Sequential(nn.Conv2d(curr_dim, z_dim, k_size))
        self.attn1 = Self_Attn(curr_dim // 2, 'relu')
        self.attn2 = Self_Attn(curr_dim, 'relu')

    def forward(self, x): 
        enable_attn = True
        p1, p2 = None, None
        for l in range(self.n_layer):
            if enable_attn and l == self.n_layer-1:
                x, p1 = self.attn1(x) 
            x = getattr(self, 'l%d'%(l+1))(x)

        if enable_attn:
            x,p2 = self.attn2(x)
        x=self.last(x)
        
        return x, p1, p2


class SelfAttnNet(nn.Module):
    def __init__(self, ch_in, ch_out, image_size, z_dim=1000):
        super(SelfAttnNet, self).__init__()

        self.enc = Discriminator(batch_size=1, ch_in=ch_in, z_dim=z_dim, image_size=image_size)
        self.dec = Generator(batch_size=1, z_dim=z_dim, ch_out=ch_out, image_size=image_size)
        self.latent = None
        self.enc_attn1, self.enc_attn2 = None, None
        self.dec_attn1, self.dec_attn2 = None, None

    def forward(self, x):
        self.latent, self.enc_attn1, self.enc_attn2 = self.enc(x)
        out, self.dec_attn1, self.dec_attn2 = self.dec(self.latent)
      
        return out

class UNetAttn(nn.Module): 
    def __init__(self, n_channels, n_classes, image_size, pretrained=None, z_dim = 1000): 
        super(UNetAttn, self).__init__()

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
        self.down1 = nn.Sequential(*enc_layers[:5])
        self.down2 = nn.Sequential(*enc_layers[5])
        self.down3 = nn.Sequential(*enc_layers[6]) 
        self.down4 = nn.Sequential(*enc_layers[7])   #[1, 512 ,7, 7] 
        self.extract = nn.Sequential(nn.Conv2d(512, z_dim, 7))
        
        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')
        
        self.latent = None 

        self.up = Generator(batch_size=1, image_size=image_size, z_dim=z_dim)

    def enc(self, x): 
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        #x3, p1 = self.attn1(x3) 
        x4 = self.down4(x3)
        #x4, p2 = self.attn2(x4)
        out = self.extract(x4)
        return out

    def dec(self, z):
        return self.up(z) 
    
    def forward(self, x): 
        self.latent = self.enc(x)
        out, _, _ =  self.dec(self.latent)
        return out 




if __name__ == "__main__": 
    import sys
    import matplotlib.pyplot as plt 

    sys.path.insert(0, '..') # '../..'
    from utils.image_io import * 

    img = prepare_image('../images/I/withgt_56_m.jpg', imsize=64)
    img_t = np_to_torch(img).cuda() 
    G = Generator(batch_size=1).cuda()
    D = Discriminator(batch_size=1).cuda()

    d_out, d_atn1, d_atn2 = D(img_t)
    g_out, g_atn1, g_atn2 = G(d_out)

    plot_image_grid("compare", [img, torch_to_np(g_out)])
