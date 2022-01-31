import os
import time
import sys

import cv2
import numpy as np
from skimage import io, segmentation
import math
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from scipy.io import loadmat,savemat
from net.unet_model_ori import *

from skimage import io
from skimage.util import img_as_float
from skimage.segmentation import slic, mark_boundaries  
 

mark_boundaries = segmentation.mark_boundaries
def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *=255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)

class UNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(UNet, self).__init__()
        self.inc = inconv(inp_dim, 64)
        self.down1 = down(64, 128)
        self.up4 = up(192, 128)
        self.dcs0 = DCS(128, 32, 3)
        self.outc = outconv(128, mod_dim2)
        self.dcs1 = DCS(mod_dim2, 32, 3)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up4(x2, x1)
        x = self.outc(x)
        x, mu = self.dcs1(x)
        return x

class DCS(nn.Module):
    '''The Deep Clustering Subnetwork (DCS).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of cluster centers.
        stage_num (int): The iteration number for EM.
    '''
    ### 定义DCS网络结构
    def __init__(self, c, k, stage_num=3):
        super(DCS, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)  #
        mu.normal_(0, math.sqrt(2. / k))  #
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 =nn.Conv2d(c, c, 1, bias=False)

        ####iteration
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)
                mu = self._l2norm(mu, dim=1)

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn #
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The inp
            ut tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

class Args(object):
    train_epoch =200 ## training iteration T ##
    mod_dim1 = 100
    mod_dim2 = 2# 100
    gpu_id =0 #0
    min_label_num = 1 #4  # if the label number small than it, break loop
    max_label_num = 256  # if the label number small than it, start to show result image.


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )
        self.emau = DCS(mod_dim2, 64, 3)

    def forward(self, x):
        x=self.seq(x)
        return x


def get_filelist(dir, Filelist):

    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)

    return Filelist


def segment(f, image_size=None, output_dir='output'):
    start_time0 = time.time()
    args = Args()
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0
    
    pathbsd, name = os.path.split(f)
    namep = os.path.splitext(name)[0]
    
    input_image_path = os.path.join(pathbsd, name)
    image = cv2.imread(input_image_path)
    if image_size is not None: 
        image = cv2.resize(image, (image_size, image_size))
    softmax = nn.Softmax(dim=1)

    '''segmentation ML'''
#    m = loadmat('./superpixel/'+namep+'.mat');
#    seglab = m["seg_lab"]
    
#    seglab = np.load('./superpixel/'+namep+'_seg.npy')
    seglab, boundary = get_superpixel(image.copy())
    seg_map=seglab
    show = mark_boundaries(image, seg_map)
    cv2.imwrite(os.path.join(output_dir,'{}_boundary.jpg'.format(namep)), boundary)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)
    model = UNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image

    '''train loop'''
    start_time1 = time.time()
    model.train()
    for batch_idx in range(args.train_epoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output1 = output
        output1 = output1[np.newaxis, :, :, :]
        output2 = output[0:1, :, :]
        croppings = (output2 > 0).float()

        prob_map = F.softmax(output, dim=0).permute(1,2,0).data.cpu().numpy()

        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)            
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to(device)

        loss = criterion(output, target) #as defined in Eq. (8)

        loss.backward()
        optimizer.step()

        '''show image''' #to print using the pesudo color;
        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:  # update show
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)
       
       # cv2.imshow("seg_pt", show)
       # cv2.waitKey(1)
        print(loss.item())
        if len(un_label) < args.min_label_num:
            break

    '''save'''
    sp=list(image.shape)
    label = im_target.reshape((sp[0],sp[1]))

    time0 = time.time() - start_time0
    time1 = time.time() - start_time1

    print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))
    cv2.imwrite(os.path.join(output_dir, "%s_seg.jpg" % (namep)), show)
    
    # save for DIP 
    np.save(os.path.join(output_dir, "%s_seg.npy" % (namep)), prob_map)
    cv2.imwrite(os.path.join(output_dir, "%s_init_b.jpg" % (namep)), image*prob_map[:,:,[0]])
    cv2.imwrite(os.path.join(output_dir, "%s_init_r.jpg" % (namep)), image*prob_map[:,:,[1]])
    return prob_map

def get_superpixel(img_bgr, output_dir="output/segments"):
    # get segments

    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype('float32')
    segments = slic(image, n_segments=200, max_iter=150, sigma=0, compactness=10)
    boundary = cv2.cvtColor( mark_boundaries(image, segments), cv2.COLOR_RGB2BGR)    
    
    return segments, boundary

def mainf():
   # dir='../images/I'
    LIst=get_filelist(dir, [])
   # get_superpixel(LIst)
    print(LIst)
    for ii in range(len(LIst)):
        segment(f)


if __name__ == '__main__':
    #run()
    mainf()

