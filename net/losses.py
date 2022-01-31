'''
Date: March,  2021
Author: Hamed Rahmani Khezri and Suhong Kim

This file is originally from "'Double DIP" (https://github.com/yossigandelsman/DoubleDIP) 
Some modifications are built to define the baselines
'''


import torch
from torch import nn
import numpy as np
from .layers import bn, VarianceLayer, CovarianceLayer, GrayscaleLayer
from .downsampler import * 
from torch.nn import functional as F

from torch.autograd import Variable
import torchvision.models as models

from skimage import feature
from utils.image_io import *

from vgg import Vgg19


def compute_normalized_dist(feat1, feat2, alpha=1.4): 
    alpha = Variable(torch.Tensor([alpha])).to(str(feat1.device))
    dist = -1.0*(torch.abs(feat1-feat2) - alpha*torch.exp(alpha)) / (alpha*alpha)
    return torch.sigmoid(torch.mean(dist)).squeeze()

def compute_channel_wise(img1_t, img2_t, func_loss): 
    return  sum([func_loss(img1_t[:, [i], :, :], img2_t[:, [i], :, :]) for i in range(img2_t.shape[1])])        

class SelfSupervisedLoss(nn.Module): 
    def __init__(self, image_t, dataset_name='places365', model_name='resnet18'): 
        super(SelfSupervisedLoss, self).__init__()
        self.device = str(image_t.device)
        self.img = image_t

        if dataset_name == 'imagenet':
            model_names= {
                'resnet18' : 'resnet18-5c106cde.pth',
                'vgg16': 'vgg16-397923af.pth',
                'vgg19': 'vgg19-dcbb9e9d.pth',
                'vgg16_bn': 'vgg16_bn-6c64b313.pth',
                'vgg19_bn': 'vgg19_bn-c79401a0.pth'}           
            model_file = os.path.join('./models', model_names[model_name])

            # download  model 
            if not os.access(model_file, os.W_OK):
                model_url ='https://download.pytorch.org/models/%s'%(model_names[model_name])            
                os.system('wget -P {} {}'.format('./models', model_url))

            ref_model = getattr(models, model_name)()
            ref_model.load_state_dict(torch.load(model_file))
        
            # load the class label
            classes = list()
            with open('./models/imagenet_classes.txt') as f:
                classes = [line.strip() for line in f.readlines()]
            classes = tuple(classes)

        elif dataset_name == 'places365': 
            if model_name == 'vgg16': 
                #model_file = './models/vgg16_places365.pth' 
                #ref_model = getattr(models, 'vgg16')()
                #state_dict = torch.load(model_file)
                #print(state_dict.keys())
                #print(ref_model.__dict__.keys())
                #ref_model.load_state_dict(torch.load(model_file))
                
                label_file = './models/categories_hybrid1365.txt'

            elif model_name.find('resnet') > -1: 
                model_file = './models/%s_places365.pth.tar'%(model_name)
                checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
                state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
                ref_model = models.__dict__[model_name](num_classes=365)
                ref_model.load_state_dict(state_dict)

                label_file = './models/categories_places365.txt'
        
            # load the class label
            classes = list()
            with open(label_file) as class_file:
                for line in class_file:
                    classes.append(line.strip().split(' ')[0][3:])
            classes = tuple(classes)

        # parameters' required_grad should be True to use loss function 
        # but don't need to pass these parameters into the optimizer 
        self.ref_model = ref_model.to(self.device)
#        for parameter in self.ref_model.parameters():
#            parameter.requires_grad = False
#        self.ref_model.eval()
#
        self.classes = classes
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)

        # find the self-supervised target from the mixed image
        s_probs, s_indices = self.get_pred(self.img)
        self.target = s_indices[0].unsqueeze(0)
        print("self-supervised target is")
        for i in range(5):
            print("  top {:d} : {} <- prob {:.3f}".format(
                  i, self.classes[s_indices[i]],s_probs[i]))

        self.visual_count = 0

    def get_pred(self, img):
        logit = self.ref_model(img)
        probs = F.softmax(logit).data.squeeze() 
        s_probs, s_indices = probs.sort(0, True) 
        return s_probs, s_indices
    
    def forward(self, x, y):
        return self.cross_entropy_loss(self.ref_model(x), self.target)

    def forward(self, x1, x2): 
        # x1 loss
        x1_img_cross = renormalize(smooth_image_torch(self.img.clone().detach() - x2.clone().detach()))
        x1_probs, x1_indices = self.get_pred(x1_img_cross) # cross constrain 
        x1_target = x1_indices[0].unsqueeze(0).detach()
        x1_loss = self.cross_entropy_loss(self.ref_model(x1), x1_target)

        x2_img_cross = renormalize(smooth_image_torch(self.img.clone().detach() - x1.clone().detach()))
        x2_probs, x2_indices = self.get_pred(x2_img_cross) # cross constrain 
        x2_target = x2_indices[0].unsqueeze(0).detach()
        if x1_target == x2_target: 
            x2_target = x2_indices[1].unsqueeze(0).detach()
        x2_loss = self.cross_entropy_loss(self.ref_model(x2), x2_target)
        
        if (self.visual_count % 500 )== 499:
            print()
            print("x1 top : {} <- prob {:.3f}".format(
                 self.classes[x1_target],x1_probs[0]))
            print("x2 top : {} <- prob {:.3f}".format(
                self.classes[x2_target],x2_probs[0]))
#            plot_image_grid("renoramlized", 
#                            [torch_to_np(x1_img_cross), torch_to_np(x2_img_cross)],
#                            output_path = './output/percepnet_selfloss')
        self.visual_count += 1

        return x1_loss + x2_loss 
        

class PerceptualLoss(nn.Module): 
    def __init__(self, model_name, connection_mask): 
        super(PerceptualLoss, self).__init__()
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
      
        for param in perceptual_model.parameters():
            param.requires_grad = False

        enc_layers = list(perceptual_model.children())
        self.layer1 = nn.Sequential(*enc_layers[:5])   #[1,64,56,56] 
        self.layer2 = nn.Sequential(*enc_layers[5])   #[1,128,28,28]
        self.layer3 = nn.Sequential(*enc_layers[6])   #[1,256,14,14] 
        self.layer4 = nn.Sequential(*enc_layers[7])   #[1,512,7,7] 
        # self.out = nn.Sequential(*enc_layers[8:]) # TODO

        # connection mask is based on Decoder so it should be reversed 
        self.connection_mask = connection_mask[-1::-1]
        self.connection_mask[-1] = True # last layer output should be always true
        self.connetion_mask = [True]* 4 # temp 

        self.compute_l1_loss = nn.L1Loss()
        self.compute_smooth_l1_loss = nn.SmoothL1Loss()
        self.compute_mse_loss = nn.MSELoss()
        self.compute_ssim_loss = SSIMLoss()
        self.compute_cos_loss = nn.CosineEmbeddingLoss()

    def _extract_features(self, img): 
        feat_list = []
        feat_list.append(self.layer1(img))
        feat_list.append(self.layer2(feat_list[-1]))
        feat_list.append(self.layer3(feat_list[-1]))
        feat_list.append(self.layer4(feat_list[-1]))
        # output = self.out(feat_list[-1])
        return feat_list 
    
    def normalize_features(self, feat):
        normalizer = torch.sqrt(torch.sum(feat**2, dim=1, keepdim=True))
        return feat / (normalizer+1e-10)

    def compute_dist(self, f1, f2): 
        f1_norm, f2_norm = self.normalize_features(f1), self.normalize_features(f2)
        dist = ((f1_norm - f2_norm)**2).sum(dim=1, keepdim=True)
        dist = dist.mean([2,3], keepdim=False).squeeze(0)
        return dist

    def _get_similar_distance(self,f1, f2): 
        # dist = torch.sigmoid(torch.mean(torch.abs(c - p))).squeeze()
        # dist = torch.mean(torch.abs(c - p)).squeeze()
        # dist = self.compute_ssim_loss(c, p)
        # dist = self.compute_cos_loss(c, p, torch.Tensor([1]).to(str(c.device)))
        # print("sim dist: ", dist)
        dist = self.compute_dist(f1, f2)
        return dist
    
    def _get_dissimilar_distance(self, f1, f2, alpha=1.4): 
        #alpha = Variable(torch.Tensor([alpha])).to(str(f1.device))
        # dist = -1.0*(torch.abs(f1-f2) - alpha*torch.exp(alpha)) / (alpha*alpha)
        dist = 1 - torch.sigmoid(self.compute_dist(f1, f2))
        #dist = 1 - self.compute_ssim_loss(f1, f2)
        # dist = self.compute_cos_loss(f1, f2, torch.Tensor([-1]).to(str(f1.device)))
        # print("dsim dist: ", dist)
        #dist = -1.0*(self.compute_dist(f1, f2) - alpha*torch.exp(alpha)) / (alpha*alpha)  
        return dist

    def forward(self, curr, prev):
        curr_feats = self._extract_features(curr)
        prev_feats = self._extract_features(prev)
        
        loss = torch.zeros([]).to(str(curr.device))
        for idx in range(1, len(curr_feats)): 
           loss = loss + self._get_dissimilar_distance(curr_feats[idx], prev_feats[idx])
        return loss

#    def forward(self, b, r,  m,  L1=0.25, L2=0.25, L3=0.25, L4=0.25):
#        # extract features from the pretrained model 
#        b_feats = self._extract_features(b)
#        r_feats = self._extract_features(r)
#        bc_feats = self._extract_features(m-r)
#        rc_feats = self._extract_features(m-b)
#        m_feats = self._extract_features(m)
#
#        loss =torch.zeros([]).to(str(b.device))
#
#        for idx in range(len(b_feats)): 
#            dist1 = self.compute_dist(b_feats[idx], bc_feats[idx])
#            dist2 = self.compute_dist(r_feats[idx], rc_feats[idx])
#
#            # compute dissimilar distance 
#            # dist3 = self.compute_dist(b1_feats[idx], r1_feats[idx])
#            # dist4 = self.compute_dist(b2_feats[idx], r2_feats[idx])
#            # print(dist1, dist2, dist3, dist4)
#            # loss = loss+ L1*dist1 + L2*dist2 - L3*dist3 - L4*dist4
#            
#            #loss = loss + self._get_dissimilar_distance(b1_feats[idx], r1_feats[idx])
#            dist3 = self.compute_dist(b_feats[idx], m_feats[idx])
#            dist4 = self.compute_dist(r_feats[idx], m_feats[idx])
#            #loss = loss - 0.5*self._get_dissimilar_distance(b1_feats[idx], r1_feats[idx])
#
#            loss = loss + 0.25*dist1 + 0.25*dist2 +  0.25*dist3 + 0.25*dist4
#        return loss 
        

class PerceptualLoss_backup(nn.Module): 
    def __init__(self, model_name, connection_mask): 
        super(PerceptualLoss, self).__init__()
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
        
        # parameters' required_grad should be True to use loss function 
        # but don't need to pass these parameters into the optimizer 
        
        enc_layers = list(perceptual_model.children())
        self.layer1 = nn.Sequential(*enc_layers[:5])   #[1,64,56,56] 
        self.layer2 = nn.Sequential(*enc_layers[5])   #[1,128,28,28]
        self.layer3 = nn.Sequential(*enc_layers[6])   #[1,256,14,14] 
        self.layer4 = nn.Sequential(*enc_layers[7])   #[1,512,7,7] 
        # self.out = nn.Sequential(*enc_layers[8:]) # TODO

        # connection mask is based on Decoder so it should be reversed 
        self.connection_mask = connection_mask[-1::-1]
        self.connection_mask[-1] = True # last layer output should be always true
        self.connetion_mask = [True]* 4 # temp 

        self.compute_l1_loss = nn.L1Loss()
        self.compute_smooth_l1_loss = nn.SmoothL1Loss()
        self.compute_mse_loss = nn.MSELoss()
        self.compute_ssim_loss = SSIMLoss()
        self.compute_cos_loss = nn.CosineEmbeddingLoss()

    def _extract_features(self, img): 
        feat_list = [img]
        feat_list.append(self.layer1(img))
        feat_list.append(self.layer2(feat_list[-1]))
        feat_list.append(self.layer3(feat_list[-1]))
        feat_list.append(self.layer4(feat_list[-1]))
        # output = self.out(feat_list[-1])
        return feat_list 
    
    def _get_similar_distance(self, c, p): 
        # dist = torch.sigmoid(torch.mean(torch.abs(c - p))).squeeze()
        # dist = torch.mean(torch.abs(c - p)).squeeze()
        # dist = self.compute_ssim_loss(c, p)
        dist = self.compute_cos_loss(c, p, torch.Tensor([1]).to(str(c.device)))
        # print("sim dist: ", dist)
        return dist
    
    def _get_dissimilar_distance(self, f1, f2, alpha=1.4): 
        alpha = Variable(torch.Tensor([alpha])).to(str(f1.device))
        # dist = -1.0*(torch.abs(f1-f2) - alpha*torch.exp(alpha)) / (alpha*alpha)
        # dist = torch.sigmoid(torch.mean(dist)).squeeze()
        # dist = 1 - self.compute_ssim_loss(f1, f2)
        dist = self.compute_cos_loss(f1, f2, torch.Tensor([-1]).to(str(f1.device)))
        # print("dsim dist: ", dist)
        return dist

    def forward(self, b1, b2, r1, r2, L1=0.5, L2=0.5, L3=0.5, L4=0.5):
        # extract features from the pretrained model 
        b1_feats = self._extract_features(b1)
        b2_feats = self._extract_features(b2)
        r1_feats = self._extract_features(r1)
        r2_feats = self._extract_features(r2)
        
        loss =torch.zeros([]).to(str(b1.device))
        # for idx in range(len(self.connection_mask)): 
        for idx in range(len(b1_feats)):
            if True and self.connection_mask[idx] == True: 
                # compute similar distance 
                dist1 = self._get_similar_distance(b1_feats[idx], b2_feats[idx])
                dist2 = self._get_similar_distance(r1_feats[idx], r2_feats[idx])

                # compute dissimilar distance 
                dist3 = self._get_dissimilar_distance(b1_feats[idx], r2_feats[idx])
                dist4 = self._get_dissimilar_distance(b2_feats[idx], r1_feats[idx])
                
                loss += L1*dist1 + L2*dist2 + L3*dist3 + L4*dist4
        
        loss = loss / self.connection_mask.count(True)

        return loss 

class LayerPriorLoss(nn.Module):
    def __init__(self, gt_b_np=None, gt_r_np=None) :
        super(LayerPriorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.pad_x = nn.ReplicationPad2d((0, 0, 1, 0))
        self.pad_y = nn.ReplicationPad2d((1, 0, 0, 0))
        self.computed_mask_l1_loss = ExtendedL1Loss()

        # generate gt mask 
        if gt_b_np is not None and gt_r_np is not None: 
            self.gt_b_edge = feature.canny(rgb_to_gray(gt_b_np)[0])
            self.gt_r_edge = feature.canny(rgb_to_gray(gt_r_np)[0])

    def forward(self, img1, img2, mixed, with_gt=False) :
        if with_gt:
            mask1 = Variable(torch.Tensor(self.gt_b_edge)).to(str(img1.device))
            mask2 = Variable(torch.Tensor(self.gt_r_edge)).to(str(img1.device))
        else: 
            # generate gradient mask (No grad_required)
            img1_gray = rgb_to_gray(np.clip(torch_to_np(img1.data), 0, 1))  # detach() makes the data in-place changed 
            img2_gray = rgb_to_gray(np.clip(torch_to_np(img2.data), 0, 1)) 
            img1_edge = feature.canny(img1_gray[0], sigma=0.1) # reduce the dimension to 2-d
            img2_edge = feature.canny(img2_gray[0], sigma=0.1)

            mask1 = Variable(torch.Tensor(img1_edge)).to(str(img1.device))
            mask2 = Variable(torch.Tensor(img2_edge)).to(str(img1.device))

        mask1 = torch.cat(tuple([mask1.unsqueeze(0).unsqueeze(0)]*img1.shape[1]), dim=1) # cat in ch (1, ch, w, h)
        mask2 = torch.cat(tuple([mask2.unsqueeze(0).unsqueeze(0)]*img2.shape[1]), dim=1)

        # get img gradient 
        img1_gradx, img1_grady = self.compute_gradient(img1)
        img2_gradx, img2_grady = self.compute_gradient(img2)
        mixed_gradx, mixed_grady = self.compute_gradient(mixed)

        # loss = self.computed_mask_l1_loss(img1_gradx, mixed_gradx, mask1)
        # loss += self.computed_mask_l1_loss(img1_grady, mixed_grady, mask1)
        # loss += self.computed_mask_l1_loss(img2_gradx, mixed_gradx, mask2)
        # loss += self.computed_mask_l1_loss(img2_grady, mixed_grady, mask2)

        # loss += self.computed_mask_l1_loss(img1_gradx, img2_gradx, mask1) 
        # loss += self.computed_mask_l1_loss(img1_grady, img2_grady, mask1) 
        # loss += self.computed_mask_l1_loss(img2_gradx, img1_gradx, mask2) 
        # loss += self.computed_mask_l1_loss(img2_grady, img1_grady, mask2) 

        loss1 = torch.sum(mask2 * torch.abs(mixed_gradx) * torch.abs(img1_gradx) + mask2 *  torch.abs(mixed_grady) * torch.abs(img1_grady)) / (torch.sum(mask2) + 1) # 1 for no-zero division 
        loss2 = torch.sum(mask1 * torch.abs(mixed_gradx) * torch.abs(img2_gradx) + mask1 *  torch.abs(mixed_grady) * torch.abs(img2_grady)) / (torch.sum(mask1) + 1)
        loss = loss1 + loss2
        return loss

    def compute_gradient(self, img):
        img_xpad = self.pad_x(img)
        img_ypad = self.pad_y(img)

        gradx = img_xpad[:, :, 1:, :] - img_xpad[:, :, :-1, :]
        grady = img_ypad[:, :, :, 1:] - img_ypad[:, :, :, :-1]

        return gradx, grady

class NormalizedL2Loss(nn.Module):
    """
    also pays attention to the mask, to be relative to its size
    """
    def __init__(self):
        super(ExtendedL1Loss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, a, b, mask):
        normalizer = self.l1(mask, torch.zeros(mask.shape).cuda())
        # if normalizer < 0.1:
        #     normalizer = 0.1
        c = self.l1(mask * a, mask * b) / normalizer
        return c

class SSIMLoss(nn.Module): 
    def __init__(self): 
        super(SSIMLoss, self).__init__()

        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y): 
        x = self.pad(x)
        y = self.pad(y)
        mu_x = self.avg_pool(x)
        mu_y = self.avg_pool(y)

        sigma_x  = self.avg_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.avg_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.avg_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d + 1e-20) / 2, 0, 1).mean()


class MS_SSIMLoss(nn.Module): 
    # https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
    def __init__(self, window_size=11, reduction=True): 
        super(MS_SSIMLoss, self).__init__()
        self.window_size = window_size 
        self.reduction = reduction
        self.level_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        
    def _gaussian(self, win_size, sigma):
        win_size_t = torch.Tensor([win_size])
        gauss = torch.Tensor([torch.exp(-(x - win_size_t//2)**2/float(2*sigma**2)) for x in range(win_size)])
        return gauss/gauss.sum()

    def _create_window(self, win_size, channel=1):
        _1D_window = self._gaussian(win_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, win_size, win_size).contiguous()
        return window
    
    def _ssim(self, img1, img2): 
        
        (_, channel, height, width) = img1.size()
        win_size = min(self.window_size, height, width)
        window = self._create_window(win_size, channel).to(str(img1.device)).type(img1.dtype)
        padd = win_size//2

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity
    
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        ssim = ssim_map.mean() if self.reduction else ssim_map.mean(1).mean(1).mean(1) 

        return ssim

    def forward(self, img1_t, img2_t): 
        return self._ssim(img1_t, img2_t)

    def forward_ms(self, img1_t, img2_t):
        # check the range of inputs 
        if torch.max(img1_t) > 2: # if 0 ~ 255
            img1_t = img1_t / 255. # normalize 
            img2_t = img2_t / 255.
        if torch.min(img1_t) < -0.0: # if -1 ~ 1 (tanh)
            img1_t = (img1_t + 1) / 2.0 # shift -1~1 to 0~1
            img2_t = (img2_t + 1) / 2.0 # shift -1~1 to 0~1

        mssim, mcs = [], []
        img1, img2 = img1_t, img2_t 
        for weight in range(len(self.level_weights)):
            ssim, cs = self._ssim(img1, img2)
            # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
            # if normalize: 
            #     ssim = (ssim + 1) / 2.0 
            #     cs = (cs + 1) / 2.0 
            mssim.append(ssim**weight)
            mcs.append(cs**weight)
            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        # mssim = torch.stack(mssim)
        # mcs = torch.stack(mcs)
        # # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        # output = torch.prod(mcs[:-1] * mssim[-1])

        output = torch.prod(torch.stack(mssim))
        return output


class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))


class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

        filt_x = np.array([[[1, 0, -1],[2,0,-2],[1,0,-1]]]*3)
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight=nn.Parameter(torch.from_numpy(filt_x).float().unsqueeze(0))
        
        filt_y = np.array([[[1, 2, 1],[0,0,0],[-1,-2,-1]]]*3)
        self.conv_y=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight=nn.Parameter(torch.from_numpy(filt_y).float().unsqueeze(0))
        
    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(grad1_s.shape[1]):
            for j in range(grad2_s.shape[1]):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
#        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
#        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
#        return gradx, grady
        self.conv_x.to(str(img.device))
        self.conv_y.to(str(img.device))
        G_x = self.conv_x(img)
        G_y = self.conv_y(img)
        return G_x, G_y


class ExtendedL1Loss(nn.Module):
    """
    also pays attention to the mask, to be relative to its size
    """
    def __init__(self):
        super(ExtendedL1Loss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, a, b, mask):

        c = self.l1(mask * a, mask * b).sum()
        return c


class NonBlurryLoss(nn.Module):
    def __init__(self):
        """
        Loss on the distance to 0.5
        """
        super(NonBlurryLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x):
        return 1 - self.mse(x, torch.ones_like(x) * 0.5)


class GrayscaleLoss(nn.Module):
    def __init__(self):
        super(GrayscaleLoss, self).__init__()
        self.gray_scale = GrayscaleLayer()
        self.mse = nn.MSELoss().cuda()

    def forward(self, x, y):
        x_g = self.gray_scale(x)
        y_g = self.gray_scale(y)
        return self.mse(x_g, y_g)


class GrayLoss(nn.Module):
    def __init__(self):
        super(GrayLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, x):
        y = torch.ones_like(x) / 2.
        return 1 / self.l1(x, y)

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

        filt_x = np.array([[[1, 0, -1],[2,0,-2],[1,0,-1]]]*3)
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight=nn.Parameter(torch.from_numpy(filt_x).float().unsqueeze(0))
        
        filt_y = np.array([[[1, 2, 1],[0,0,0],[-1,-2,-1]]]*3)
        self.conv_y=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight=nn.Parameter(torch.from_numpy(filt_y).float().unsqueeze(0))
    
    def forward(self, x, mean=False):
        self.conv_x.to(str(x.device))
        self.conv_y.to(str(x.device))
        G_x = self.conv_x(x)
        G_y = self.conv_y(x)
        
        if mean: 
            loss  = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
            return torch.mean(loss.view(-1))
        else:
            return G_x, G_y


class GradientLoss_classic(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(GradientLoss_classic, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)


class YIQGNGCLoss(nn.Module):
    def __init__(self, shape=5):
        super(YIQGNGCLoss, self).__init__()
        self.shape = shape
        self.var = VarianceLayer(self.shape, channels=1)
        self.covar = CovarianceLayer(self.shape, channels=1)

    def forward(self, x, y):
        if x.shape[1] == 3:
            x_g = rgb_to_yiq(x)[:, :1, :, :]  # take the Y part
            y_g = rgb_to_yiq(y)[:, :1, :, :]  # take the Y part
        else:
            assert x.shape[1] == 1
            x_g = x  # take the Y part
            y_g = y  # take the Y part
        c = torch.mean(self.covar(x_g, y_g) ** 2)
        vv = torch.mean(self.var(x_g) * self.var(y_g))
        return c / vv


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


# VGG loss for IBS component 
        

class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()        
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        
        return loss


## Joint generate and separate



