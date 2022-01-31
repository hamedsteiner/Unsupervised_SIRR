############ Dialated: H3
## added file name for saving images

##

## For Exp3


'''
Date: Jan,  2021 
Author: Hamed Rahmnai
'''

from collections import namedtuple
import time
import numpy as np
from skimage import exposure

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

# local 
from net.percep_model_h3 import *
from net.losses import *
from net.noise import get_noise
from utils.image_io import *

im_size = 224
class Separator:
    def __init__(self, folder_name, image_file, 
                    # image info 
                    image_gt_b_file=None, image_gt_r_file=None,
                    image_weak_b_file = None, image_size=im_size,            
                    # structure info
                    architecture='percep2', pretrained='places365',
                    input_type='image', alpha_gt=0.4, 
                    enable_cross=False, enable_feedback=False, 
                    enable_augment=False, enable_alpha=False,
                    # Loss Params
                    recon_loss_weight=0.0, gray_loss_weight=0.0,
                    cross_loss_weight=0.0, excld_loss_weight=0.0, 
                    smooth_loss_weight=0.0, ceil_loss_weight=0.0,
                    tvex_loss_weight=0.0, weak_weight = 0.0,
                    vgg_weight = 0.0, 
                    # training info
                    num_iter=5000, learning_rate=0.0001, 
                    show_every=100, plot_every=-1, 
                    outdir='./output', outdir_name=None, 
                    model_init_dir=None,   
                 ):

            # -- Arguments--
        self.folder_name = folder_name
        self.image_file = image_file 
        self.image_gt_b_file = image_gt_b_file
        self.image_gt_r_file = image_gt_r_file
        self.image_weak_b_file = image_weak_b_file

        self.alpha_gt = alpha_gt
        self.image_size = image_size
        self.architecture = architecture
        self.pretrained = pretrained
        self.input_type = input_type
        self.enable_cross = enable_cross
        self.enable_feedback = enable_feedback
        self.enable_augment = enable_augment
        self.enable_alpha = enable_alpha
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.plot_every = plot_every       
        self.show_every = show_every      
        self.model_init_dir = model_init_dir
        # init for Losses 
        self.recon_loss_weight = recon_loss_weight      #"default" or "chwise"
        self.gray_loss_weight = gray_loss_weight    # gray-scale recon loss
        self.cross_loss_weight = cross_loss_weight
        self.excld_loss_weight = excld_loss_weight  # exclusion loss (from DoubleDIP)
        self.smooth_loss_weight = smooth_loss_weight
        self.ceil_loss_weight = ceil_loss_weight    # image reflection prior loss
        self.tvex_loss_weight = tvex_loss_weight

        self.weak_weight = weak_weight # for weak supervision

        self.vgg_weight = vgg_weight



        # --Environment--
        if not os.path.exists(outdir): 
            os.makedirs(outdir)
        if outdir_name == None: 
            outdir_name = "{}_{}_{}".format(architecture, 
                                            input_type, 
                                            time.strftime("%Hh%Mmin",time.localtime()))
       
        self.output_path = os.path.join(outdir, outdir_name)
        if not os.path.exists(self.output_path): 
            os.makedirs(self.output_path)

        print(">>>>>>>>>>>>> The output will be saved in", self.output_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(">>>>>>>>>>>>> This model will be trained on", self.device)
    
        # --Type definition--
        self.LOSSES = namedtuple('LOSSES', ['loss', 'recon', 'gray', 'cross', 'excld','smooth', 'ceil', 'weak', 'tvex', 'vgg'])
        self.PSNRS  = namedtuple('PSNRS',  ['psnr', 'psnr_pair1', 'psnr_pair2'])
        self.SSIMS  = namedtuple('SSIMS',  ['ssim', 'ssim_pair1', 'ssim_pair2'])
        self.RESULT = namedtuple('RESULT', ['reflection', 'background', 'reconstruction', 'losses', 'psnrs', 'ssims'])


    def run(self, num_run, in_batch=False, show=True):    
        best_list = []
        n, cnt = 0, 0
        while(n < num_run):
            print(">>>>>>>>>>>>> running ({:d} / {:d}) on ....".format(n+1, num_run))
            self.initialize()
            status = self.optimize()
            if not status:
                cnt += 1
                if cnt > num_run: 
                    print("************* Stop running due to NaN loss *************")
                    break
                print("************* Loss reached to NaN, restarted *************")
                n = max(n-1, 0)
                continue
            self.finalize(n, show) 
            best_list.append(self.best_result)
            n = n + 1
        
        self.finalize_all(best_list) 

    def initialize(self, num_batch=1): 
        # initialize all global vairables 
        self.loss_list = []
        self.psnr_list = []
        self.ssim_list = []
        self.best_result = None

        self._init_image()
        self._init_inputs(num_batch)
        self._init_nets()
        self._init_losses()

    def _init_image(self):
        # get image name
        self.image_name = os.path.basename(self.image_file).split('.')[0]
        
        # open images as numpy array
        
        # f = loadmat(AB_path)
        # # print(f.keys())
        # AB = f.get('PixInp_B')
        # AB = np.array(AB) # For converting to a NumPy array
        # AB = AB.astype(np.float64)
        self.image_np   = prepare_image(self.image_file, imsize=self.image_size)
        self.image_gt_b = prepare_image(self.image_gt_b_file, imsize=self.image_size) if self.image_gt_b_file is not None else None
        self.image_gt_r = prepare_image(self.image_gt_r_file, imsize=self.image_size) if self.image_gt_r_file is not None else None
        
        self.image_weak_b_np = prepare_image(self.image_weak_b_file, imsize=self.image_size) if self.image_weak_b_file is not None else None
        

        print(self.image_np.shape)
        print(type(self.image_np))
        # exit()
        
        if self.image_gt_r is None and self.image_gt_b is not None:
            self.image_gt_r = self.image_np - self.image_gt_b 

        # convert image into tensor 
        self.image_t = np_to_torch(self.image_np).to(self.device) 
        self.image_gt_b_t = np_to_torch(self.image_gt_b).to(self.device)
        self.image_gt_r_t = np_to_torch(self.image_gt_r).to(self.device)

        self.image_weak_b = np_to_torch(self.image_weak_b_np).to(self.device)


        

        # print(self.image_t.shape)
        # print(type(self.image_t))
        # exit()

        # print(self.image_t)
        # exit()

    def _init_inputs(self, num_batch=1):
        if self.input_type == 'noise' or self.input_type == 'meshgrid':
            self.dipnet_b_input = get_noise(self.image_np.shape[0], self.input_type, (self.image_np.shape[1], self.image_np.shape[2])).to(self.device)
            self.dipnet_r_input = get_noise(self.image_np.shape[0], self.input_type, (self.image_np.shape[1], self.image_np.shape[2])).to(self.device)
        else: 
            # if the input type is an image 
            self.dipnet_b_input = self.image_t
            self.dipnet_r_input = self.image_t
        
        # # if num_batch is more than 1 
        self.dipnet_b_input = torch.cat(tuple([self.dipnet_b_input]*num_batch), 
                                        dim=len(self.dipnet_b_input.shape)-1)
        self.dipnet_r_input = torch.cat(tuple([self.dipnet_r_input]*num_batch), 
                                        dim=len(self.dipnet_r_input.shape)-1)

        

    def _init_nets(self) :
        num_ch_in_b = self.dipnet_b_input.shape[1]
        num_ch_in_r = self.dipnet_r_input.shape[1]
        num_ch_out = self.image_np.shape[0]


        # print(num_ch_in_r)
        # print(num_ch_out)
        # exit()

        # select the type of net structure
        dipnet_b = PercepNet(num_ch_in_b, num_ch_out, self.image_t.clone(), self.pretrained, enable_attn=False)
        dipnet_r = PercepNet(num_ch_in_r, num_ch_out, self.image_t.clone(), self.pretrained, enable_attn=False)
        
        # assign nets to the system 
        self.dipnet_b = dipnet_b.to(self.device)
        self.dipnet_r = dipnet_r.to(self.device)
        
        if self.enable_alpha: 
           # alphanet = AlphaNet(num_ch_in_b, num_ch_out, self.image_t.clone(), self.pretrained)
           # self.alphanet = alphanet.to(self.device)
            self.alphanet = Variable(self.alpha_gt*torch.ones(1,1).to(self.device), requires_grad=True)

        # initialize 
        if self.model_init_dir is not None: 
            try: 
                checkpoint = torch.load(self.model_init_dir)
                self.dipnet_b.load_state_dict(checkpoint['dipnet_b'])
                self.dipnet_r.load_state_dict(checkpoint['dipnet_r'])
                print("loadded")
            except: 
                print("fail to load init model")

        # assign parameters for the optimizer
        self.parameters = [p for p in self.dipnet_b.parameters() if p.requires_grad] + \
                          [p for p in self.dipnet_r.parameters() if p.requires_grad]
        
        if self.enable_alpha: 
            #self.parameters += [p for p in self.alphanet.parameters() if p.requires_grad]
            self.parameters += [self.alphanet]
            
        # Compute number of parameters
        s  = sum(np.prod(list(p.size())) for p in self.parameters)
        print ('------------> Number of params: {:,}'.format(s))







    def _init_losses(self):
        self.compute_l1_loss = nn.L1Loss().to(self.device)
        self.compute_smooth_l1_loss = nn.SmoothL1Loss().to(self.device)
        self.compute_mse_loss = nn.MSELoss().to(self.device)       
        self.compute_ssim_loss = SSIMLoss().to(self.device)

        self.compute_exclusion_loss =  ExclusionLoss().to(self.device)
        self.compute_mask_l1_loss = ExtendedL1Loss().to(self.device)
        self.compute_gradient_loss = GradientLoss().to(self.device) 

        self.compute_vgg_loss = VGGLoss().to(self.device)
    






    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.AdamW(self.parameters, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5) 
       # optimizer = torch.optim.RMSprop(self.parameters, lr=self.learning_rate, momentum=0.9)
        for itr in range(self.num_iter):        
            # optimization 
            itr_begin = time.time()
            optimizer.zero_grad()
            self._optimize_on_iter(itr)
            optimizer.step()
            scheduler.step()
            itr_end = time.time()
            self._harvest_results(itr)
            
            # post-processing 
            # display results
            self._print_status(itr, (itr_end-itr_begin))
            self._plot_results(itr)
    
            # check return status
            if np.isnan(self.current_result.losses.loss):
                return False 
            if self.current_result.psnrs.psnr > 40.0: 
                print("-----------Early Stopping") 
                return True 
        return True

    def _optimize_on_iter(self, step):
        # process
        if self.enable_feedback and step > 0:
            b_input = self.background_out_prev      
            r_input = self.reflection_out_prev 
            
            if self.enable_cross:
                b_input = self.background_cross  
                r_input = self.reflection_cross  
            
        else:
            b_input = self.dipnet_b_input
            r_input = self.dipnet_r_input
        
        if self.enable_augment and self.input_type != 'image_feat':
            transform = transforms.Compose([
                            transforms.ColorJitter(
                                brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05), 
                            transforms.ToTensor()
                        ])
            b_input = transform(np_to_pil(torch_to_np(b_input))).unsqueeze(0).to(self.device) 
            r_input = transform(np_to_pil(torch_to_np(r_input))).unsqueeze(0).to(self.device)
      
        if self.enable_alpha:
            self.alpha_out =  self.alphanet
            self.background_pure = self.dipnet_b(b_input)
            self.reflection_pure = self.dipnet_r(r_input)
            self.background_out = (1-self.alpha_out)*self.background_pure
            self.reflection_out = self.alpha_out*self.reflection_pure
            #self.background_out = (1-self.alpha_out)*self.dipnet_b(b_input)
            #self.reflection_out = self.alpha_out*self.dipnet_r(r_input)
            self.recon_out = self.background_out + self.reflection_out


            self.background_cross = self.image_t.clone().detach() - self.reflection_out.clone().detach()

            
            self.reflection_cross = self.image_t.clone().detach() - self.background_out.clone().detach()
            
        else: 
            self.background_out = self.dipnet_b(b_input)
            self.reflection_out = self.dipnet_r(r_input)
            self.recon_out = self.background_out + self.reflection_out
            self.background_cross = self.image_t.clone().detach() - self.reflection_out.clone().detach()
            self.reflection_cross = self.image_t.clone().detach() - self.background_out.clone().detach()
        
        residue_mean = torch.mean(torch.mean(self.reflection_cross.squeeze(0), dim=2),dim=1)
        residue_mean = torch.cat([residue_mean[ch].repeat(self.image_size, self.image_size).unsqueeze(0) for ch in range(len(residue_mean))], dim=0)
        residue_mean = residue_mean.unsqueeze(0)     
        self.background_cross = torch.clamp(self.background_cross + residue_mean, 0., 1.)
        self.reflection_cross = torch.clamp(self.reflection_cross - residue_mean, 0., 1.) 

        # when the net has latent
        try:
            self.latent_b_out = self.dipnet_b.latent
            self.latent_r_out = self.dipnet_r.latent
        except:
            self.latent_b_out = None 
            self.latent_r_out = None 

        # compute loss
        total_loss = self._compute_losses(step)

        # backpropagate 
        total_loss.backward()

        # update prev
        self.background_out_prev = self.background_out.clone().detach().requires_grad_(True)
        self.reflection_out_prev = self.reflection_out.clone().detach().requires_grad_(True)
   
    def _compute_losses(self, step):
        # initilize loss buffer
        loss_out = dict.fromkeys(self.LOSSES._fields,torch.zeros((1), 
                                 device=self.device, requires_grad=True))

        # ---reconstruction loss
        loss_out['recon'] = loss_out['recon'] + self.recon_loss_weight \
                            * self.compute_mse_loss(self.recon_out, self.image_t)
        if step < 1000: 
            gt_gx, gt_gy = self.compute_gradient_loss(self.image_t)
            gx, gy = self.compute_gradient_loss(self.recon_out)
            loss_out['recon'] = loss_out['recon'] \
                                + 0.07*self.recon_loss_weight * self.compute_l1_loss(gx, gt_gx) \
                                + 0.07*self.recon_loss_weight * self.compute_l1_loss(gy, gt_gy)
        if step > 0:
            # self-recon loss 
            loss_out['recon'] = loss_out['recon'] + 0.1*self.recon_loss_weight \
                                * self.compute_ssim_loss(self.background_out,
                                                         self.background_out_prev) 
            
            loss_out['recon'] = loss_out['recon'] + 0.1*self.recon_loss_weight \
                                * self.compute_ssim_loss(self.reflection_out, 
                                                         self.reflection_out_prev) 

        # ---gray-scale loss
        loss_out['gray'] = loss_out['gray'] + self.gray_loss_weight \
                           * self.compute_mse_loss(rgb_to_gray(self.recon_out), 
                                                   rgb_to_gray(self.image_t))
        
       # ---exclusion loss
        loss_out['excld'] = loss_out['excld'] + self.excld_loss_weight \
                            * self.compute_exclusion_loss(self.background_out, self.reflection_out)
        
        # ---Total Variance Balance loss (total variance should be balanced) 
        loss_out['tvex'] = loss_out['tvex'] +  self.tvex_loss_weight \
                             * torch.abs(self.compute_gradient_loss(self.background_out, mean=True) 
                                       - self.compute_gradient_loss(self.reflection_out, mean=True))
        
        # ---smooth loss(Total Variance Loss) 
        loss_out['smooth'] = loss_out['smooth'] + 0.5 * self.smooth_loss_weight \
                             * self.compute_gradient_loss(self.background_out, mean=True)
        loss_out['smooth'] = loss_out['smooth'] + 0.5 * self.smooth_loss_weight \
                             * self.compute_gradient_loss(self.reflection_out, mean=True)

        # ---ceil loss 
        loss_out['ceil'] = loss_out['ceil'] + 0.5 * self.ceil_loss_weight \
                           * self.compute_mask_l1_loss(self.background_out, self.image_t, 
                                                       (self.background_out > self.image_t))
        loss_out['ceil'] = loss_out['ceil'] + 0.5 * self.ceil_loss_weight \
                           * self.compute_mask_l1_loss(self.reflection_out, self.image_t, 
                                                       (self.reflection_out > self.image_t))   
        
        # ---cross-guiding loss 
        if self.cross_loss_weight > 0 and step > 0: 
            loss_out['cross'] = loss_out['cross'] + 0.5 * self.cross_loss_weight \
                                * self.compute_mse_loss(self.background_out, 
                                                       self.background_cross)  
            loss_out['cross'] = loss_out['cross'] + 0.5 * self.cross_loss_weight \
                                * self.compute_mse_loss(self.reflection_out,
                                                       self.reflection_cross)  






        # -- Weak Supervision( For now just Background) 0.5 * B + 0.5 * R for future

        # -- Cross or out?

        loss_out['weak'] = loss_out['weak'] + 1.0 * self.weak_weight \
                           * self.compute_mse_loss(self.background_out, self.image_weak_b)



        # -- VGG19 loss

        # -- coeff increasing ? 

        loss_out['vgg'] = loss_out['vgg'] + 1.0 * self.weak_weight \
                           * self.compute_vgg_loss(self.background_out, self.image_weak_b)
        









        
        # compute total 
        t_loss = sum(loss_out.values())
        
        # save loss_out  
        self.losses = self.LOSSES(loss=t_loss.item(),            recon=loss_out['recon'].item(), 
                                  gray=loss_out['gray'].item(),  cross=loss_out['cross'].item(),  
                                  excld=loss_out['excld'].item(),smooth=loss_out['smooth'].item(), 
                                  ceil=loss_out['ceil'].item(),  weak=loss_out['weak'].item(),
                                  tvex=loss_out['tvex'].item(),  vgg=loss_out['vgg'].item())
        
        # return 
        return t_loss 

    def _harvest_results(self, step):
        """
        All the results here should be separated from the graph 
        """
        # network results
        background_out_np = np.clip(torch_to_np(self.background_out.data), 0, 1)
        reflection_out_np = np.clip(torch_to_np(self.reflection_out.data), 0, 1)
        recon_out_np      = np.clip(torch_to_np(self.recon_out.data), 0, 1)
        self.loss_list.append(self.losses.loss) 

        # evaluation results
        # PSNR
        psnr = compute_psnr(self.image_np, recon_out_np)
        self.psnr_list.append(psnr)
        # for speed, cacluate other psnrs every show_every 
        if step == 0 or step % self.show_every == self.show_every - 1 :
            psnr_b = compute_psnr(self.image_gt_b, background_out_np)
            psnr_r = compute_psnr(self.image_gt_r, reflection_out_np)
            self.psnr_pair1 = (psnr_b, psnr_r)
            psnr_b = compute_psnr(self.image_gt_r, background_out_np)
            psnr_r = compute_psnr(self.image_gt_b, reflection_out_np)
            self.psnr_pair2 = (psnr_b, psnr_r)
        # save panrs
        psnrs = self.PSNRS(psnr, self.psnr_pair1, self.psnr_pair2)
        
        # SSIM 
        ssim = compute_ssim(self.image_np, recon_out_np)
        self.ssim_list.append(ssim)
        # for speed, cacluate other ssims every show_every 
        if step == 0 or step % self.show_every == self.show_every - 1 :
            ssim_b = compute_ssim(self.image_gt_b, background_out_np)
            ssim_r = compute_ssim(self.image_gt_r, reflection_out_np)
            self.ssim_pair1 = (ssim_b, ssim_r)
            ssim_b = compute_ssim(self.image_gt_r, background_out_np) 
            ssim_r = compute_ssim(self.image_gt_b, reflection_out_np)
            self.ssim_pair2 = (ssim_b, ssim_r)
        # save ssims   
        ssims = self.SSIMS(ssim, self.ssim_pair1, self.ssim_pair2) 

        # update the current result
        self.current_result = self.RESULT(background=background_out_np, 
                                          reflection=reflection_out_np, 
                                          reconstruction=recon_out_np, 
                                          losses=self.losses, 
                                          psnrs=psnrs, ssims=ssims)
        # update the best result
        if self.best_result is None or self.best_result.psnrs.psnr < self.current_result.psnrs.psnr:
            self.best_result = self.current_result
            # save model 
#            torch.save({"architecture": self.architecture, 
#                        "connection": self.connection, 
#                        "dipnet_b": self.dipnet_b.state_dict(), 
#                        "dipnet_r": self.dipnet_r.state_dict()}, os.path.join(self.output_path, "checkpoint"))
    
    def _print_status(self, step, duration):
        if step % self.show_every == self.show_every - 1 :
            loss_str = "Loss:"
            for name, value in zip(self.LOSSES._fields, self.current_result.losses): 
                if value == 0.0: 
                    continue
                if name == 'loss': 
                    loss_str += "{:f} (".format(value)
                else:
                    loss_str += " {}: {:f} ".format(name, value)
            loss_str += ")"

            psnr_str = "PSNR: {:3.3f}".format(self.current_result.psnrs.psnr)
            if self.current_result.psnrs.psnr_pair1[0] > 0: 
                psnr_str += "({:3.3f}, {:3.3f})".format(self.current_result.psnrs.psnr_pair1[0], self.current_result.psnrs.psnr_pair1[1])
                psnr_str += "({:3.3f}, {:3.3f})".format(self.current_result.psnrs.psnr_pair2[0], self.current_result.psnrs.psnr_pair2[1])

            ssim_str = "SSIM: {:1.3f}".format(self.current_result.ssims.ssim)
            if self.current_result.ssims.ssim_pair1[0] > 0: 
                ssim_str += "({:1.3f}, {:1.3f})".format(self.current_result.ssims.ssim_pair1[0], self.current_result.ssims.ssim_pair1[1])
                ssim_str += "({:1.3f}, {:1.3f})".format(self.current_result.ssims.ssim_pair2[0], self.current_result.ssims.ssim_pair2[1])
            
            if self.enable_alpha:
                alpha_str = "alpha:{:1.3f}".format(self.alpha_out.item())
            else: 
                alpha_str = ""
            
            ### print all 
            print('Iteration:{:6d} {} {} {} {} Duration: {:.4f}'.format(step, alpha_str, loss_str, psnr_str, ssim_str, duration), '\r', end='')

    def _plot_results(self, step):
        if self.plot_every > 0 and (step % self.plot_every == self.plot_every - 1) :
            plot_image_grid("{}_results_iter{}".format(self.image_name, step), 
                            [self.current_result.background, self.current_result.reflection, self.current_result.reconstruction],
                             output_path=self.output_path)
            if self.enable_alpha:  
                plot_image_grid("{}_results_pure_iter{}".format(self.image_name, step), 
                                [torch_to_np(self.background_pure), 
                                 torch_to_np(self.reflection_pure), 
                                 torch_to_np(self.background_pure + self.reflection_pure)],
                                 output_path=self.output_path)
                
            plot_image_grid("{}_results_renorm_iter{}".format(self.image_name, step), 
                            [renormalize(self.current_result.background), 
                             renormalize(self.current_result.reflection), 
                             renormalize(self.current_result.reconstruction)], 
                             output_path=self.output_path) 
            
            save_image("{}_{}_background_{:d}".format(self.folder_name, self.image_name, step), self.current_result.background, output_path=self.output_path, show=False)
            save_image("{}_{}_reflection_{:d}".format(self.folder_name, self.image_name, step), self.current_result.reflection, output_path=self.output_path, show=False) 
            save_image("{}_{}_background_cross_{:d}".format(self.folder_name, self.image_name, step), np.clip(torch_to_np(self.background_cross),0,1), output_path=self.output_path, show=False)
            save_image("{}_{}_reflection_cross_{:d}".format(self.folder_name, self.image_name, step), np.clip(torch_to_np(self.reflection_cross),0,1), output_path=self.output_path, show=False)

    
    def finalize(self, num_run, show):         
        alpha = self.alpha_out.item() if self.enable_alpha else -1


        ## --added folder_name to the beginning
        plot_image_grid("{}_{}_results_best_{:d}_a_{:1.3f}_p_{:3.2f}_{:3.2f}_s_{:1.4f}_{:1.4f}".format(
                            self.folder_name, self.image_name, num_run, alpha, self.best_result.psnrs.psnr_pair1[0], self.best_result.psnrs.psnr_pair1[1], 
                            self.best_result.ssims.ssim_pair1[0], self.best_result.ssims.ssim_pair1[1]), 
                        [self.best_result.background, self.best_result.reflection, self.best_result.reconstruction], 
                         output_path=self.output_path, show=show)   

        plot_image_grid("{}_{}_results_best_renorm_{:d}_a_{:1.3f}_p_{:3.2f}_{:3.2f}_s_{:1.4f}_{:1.4f}".format(
                            self.folder_name, self.image_name, num_run, alpha, self.best_result.psnrs.psnr_pair2[0], self.best_result.psnrs.psnr_pair2[1], 
                            self.best_result.ssims.ssim_pair2[0], self.best_result.ssims.ssim_pair2[1]), 
                        [renormalize(self.best_result.background), 
                         renormalize(self.best_result.reflection),
                         renormalize(self.best_result.reconstruction)], 
                         output_path=self.output_path, show=show)    

        save_graph("{}_loss_{:d}".format(self.image_name, num_run), self.loss_list, output_path=self.output_path, show=False)
        save_graph("{}_psnr_{:d}".format(self.image_name, num_run), self.psnr_list, output_path=self.output_path, show=False)
        save_graph("{}_ssim_{:d}".format(self.image_name, num_run), self.ssim_list, output_path=self.output_path, show=False)
        save_image("{}_{}_background_{:d}".format(self.folder_name, self.image_name, num_run), self.best_result.background, output_path=self.output_path, show=False)
        save_image("{}_{}_reflection_{:d}".format(self.folder_name, self.image_name, num_run), self.best_result.reflection, output_path=self.output_path, show=False) 
        save_image("{}_{}_background_cross_{:d}".format(self.folder_name, self.image_name, num_run), self.image_np - self.best_result.reflection, output_path=self.output_path, show=False)
        save_image("{}_{}_reflection_cross_{:d}".format(self.folder_name, self.image_name, num_run), self.image_np - self.best_result.background, output_path=self.output_path, show=False)

    def finalize_all(self, b_list):
        print(">>>>>>>>>>>>> Final Results >>>>>>>>>>>>>")
        best_t_list = [b.background for b in b_list]
        best_r_list = [b.reflection for b in b_list]
        best_m_list = [b.reconstruction for b in b_list]
        best_l_list = [b.losses.loss for b in b_list]

        # plot gt inputs if exist
        if self.image_gt_b is not None and self.image_gt_r is not None:
            plot_image_grid("{}_ground_truth".format(self.image_name), [self.image_gt_b, self.image_gt_r, self.image_np],
                            output_path=self.output_path)       
        # print the Final model with the lowest loss 
        fidx = np.argmin(best_l_list)
        plot_image_grid("{}_results_best".format(self.image_name), 
                        [best_t_list[fidx], best_r_list[fidx], best_m_list[fidx]], 
                         output_path=self.output_path)

        # print the adaptive model with the lowest loss 
        fidx = np.argmin(best_l_list)
        p2, p98 = np.percentile(best_t_list[fidx].transpose(1,2,0), (2, 98))
        equl_t = exposure.rescale_intensity(best_t_list[fidx].transpose(1,2,0), in_range=(p2, p98)).transpose(2,0,1)
        p2, p98 = np.percentile(best_r_list[fidx].transpose(1,2,0), (2, 98))
        equl_r = exposure.rescale_intensity(best_r_list[fidx].transpose(1,2,0),in_range=(p2, p98)).transpose(2,0,1)
        p2, p98 = np.percentile(best_m_list[fidx].transpose(1,2,0), (2, 98))
        equl_m = exposure.rescale_intensity(best_m_list[fidx].transpose(1,2,0), in_range=(p2, p98)).transpose(2,0,1)
        plot_image_grid("{}_results_rescaled".format(self.image_name), 
                        [equl_t, equl_r, equl_m], output_path=self.output_path)       
        
        # print the min model 
        plot_image_grid("{}_results_renorm".format(self.image_name), 
                        [renormalize(best_t_list[fidx]), renormalize(best_r_list[fidx]), renormalize(best_m_list[fidx])], 
                         output_path=self.output_path) 

       
def get_filelist(dir, Filelist=[]):
	newDir = dir
	if os.path.isfile(dir):
		Filelist.append(dir)
	elif os.path.isdir(dir):
		for s in os.listdir(dir):
			newDir = os.path.join(dir, s)
			get_filelist(newDir, Filelist)

	return Filelist

     
#    functions to be added 
#    1) compare the histogram of the image -> compare with the output 
#        -> adjusting the value based on flickering removal 
#    2) resize the image as its original 
    

if __name__ == "__main__":


    # mahsa = get_filelist("./images_analyze")
    # mahsa = get_filelist("./images_new")
    
    # mahsa = os.listdir("./Kaggle_image/I_224")
    mahsa = os.listdir("./Data/Exp3_SIR")
    
    # get_image_file = lambda n : '{}/{}'.format('./Huawei_I', n)

    get_weak_file = lambda n : 'Data/Exp3_SIR/{}/{}'.format(n, 't_output.png')


    get_image_file = lambda n : 'Data/Exp3_SIR/{}/{}'.format(n, 'input.png')


    # print((mahsa))
    # exit()
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)  
    
    # ----- Samples -------#
    image_dir = 'images'
    #names = ['withgt_87',  'wild', 'Real20_9'] 
    names = [ 'toy1', 'toy3', 'toy2', 'toy', 'sample1']
    names = [ 'input1', 'input2']
    image_dir = 'Kaggel_image'
    # get_image_file = lambda n : '{}/I/{}'.format(image_dir, n)
    # get_image_gt_b_file = lambda n :'{}/B/{}'.format(image_dir, n)
    # image_dir = './Kaggle_image'


    # get_image_file = lambda n : '{}/I_400/{}'.format(image_dir, n)
    # get_image_gt_b_file = lambda n :'{}/B_400/{}'.format(image_dir, n)

    # get_image_file = lambda n : '{}/I_224/{}'.format(image_dir, n)
    # get_image_gt_b_file = lambda n :'{}/B_224/{}'.format(image_dir, n)

#    get_image_gt_r_file = lambda n :'{}/R/{}_r.jpg'.format(image_dir, n)

    # -------- SIR ----------# 
    # image_dir = 'images_new'
    # names_m = get_filelist(os.path.join(image_dir, 'I'), []) 

    # names = sorted([os.path.basename(f)[:-6] for f in names_m])
    """
    TO DO
    """

    ########EVALUATION ############################################################
    # names = names[:-8]
    # print(len(names))
    # exit()

    # if True: 
    #     best = ['9','87','105']#,'9', '87, '2', '61', '56', '37']
    #     # names = best
    # names.remove('9')
    # names.remove('87')
    # names.remove('105')
    # # print(len(names))
    # # exit()
    # get_image_file = lambda n : '{}/I/{}_m.jpg'.format(image_dir, n)
    # get_image_gt_b_file = lambda n :'{}/B/{}_g.jpg'.format(image_dir, n)
    # get_image_gt_r_file = lambda n :'{}/R/{}_r.jpg'.format(image_dir, n)

    # # -------Berkely -----# 
    # image_dir = 'images_zhang'
    # names_m = get_filelist(os.path.join(image_dir, 'blended'), []) 
    # names = sorted([os.path.basename(f)[:-4] for f in names_m])
    # names = ['15'] # 25, 29, 93, 107
    # get_image_file = lambda n : '{}/blended/{}.jpg'.format(image_dir, n)
    # get_image_gt_b_file = lambda n :'{}/transmission_layer/{}.jpg'.format(image_dir, n)
    # get_image_gt_r_file= lambda n : None

    # mahsa = mahsa[0:1]
    alpha_vals = [0.25]
    for alpha_val in alpha_vals:
        # for idx, name in enumerate(names):
        for idx, name in enumerate(mahsa):
            # print(get_image_file(name))  

            # exit()

            print("*********************************")
            print("running on {} ({:d}/{:d})".format(name,idx+1, len(names)))
            print("*********************************")
            model3 = Separator(

                    folder_name = name, 
                    image_file=get_image_file(name),
                    # image_gt_b_file=name,
                    # image_gt_b_file=get_image_gt_b_file(name),
                    # image_file=name,
                    image_gt_b_file=get_image_file(name),
                    # # image_gt_b_file=name,
                    # image_gt_b_file=get_image_gt_b_file(name),
                    # image_gt_r_file=get_image_gt_r_file(name),
                    # image_gt_b_file=None,
                    image_gt_r_file=None,

                    image_weak_b_file = get_weak_file(name),
                    # exp1 parameters -----------
                    image_size=im_size,
                    pretrained = 'places365',
                    input_type = 'image',
                    enable_cross = True, #True,
                    enable_feedback = True, # True,
                    enable_alpha = True, #True,
                    enable_augment = True, # helping to stabilize the graph?  
                    alpha_gt = alpha_val, # 0.1
                    # exp2 parameters -----------
                    recon_loss_weight = 1.2, 
                    gray_loss_weight  = 0.13, 
                    cross_loss_weight = 0.13, #0.1,
                    excld_loss_weight = 0.1,
                    ceil_loss_weight  = 0.9, # 1 
                    smooth_loss_weight= 0.005, 
                    tvex_loss_weight  = 0.001, 
                    weak_weight = 0.55,
                    vgg_weight = 0.1,

                    # training parameters ------- 
                    num_iter= 9500, learning_rate=0.0001,#0.0001 
                    show_every=500, plot_every=5000, 
                    outdir = './Output/',
                    
                    # outdir_name = 'ALL_SIR_Dialated_BjozGhablia{}'.format(alpha_val))
                    # outdir_name = 'kaggle_images2{}'.format(alpha_val))
                    outdir_name = 'Exp4_alaki_3_{}'.format(alpha_val))
                    #  outdir_name = 'H3{}'.format(alpha_val))
            model3.run(num_run=2)
            del model3


"""
 recon_loss_weight = 1.5, 
                    gray_loss_weight  = 0.13, 
                    cross_loss_weight = 0.13, #0.1,
                    excld_loss_weight = 0.1,
                    ceil_loss_weight  = 0.8, # 1 
                    smooth_loss_weight= 0.005, 
                    tvex_loss_weight  = 0.001, 
                    weak_weight = 0.55,
                    vgg_weight = 0.1,

                    # training parameters ------- 
                    num_iter= 4000, learning_rate=0.0001,#0.0001 
                    show_every=500, plot_every=5000, 
                    outdir = './Output/',

"""