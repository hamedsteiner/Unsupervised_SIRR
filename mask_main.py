 # -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:29:53 2021

@author: hr1
"""
import cv2
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import numpy.ma as ma

#print(os.listdir())
img = cv2.imread("59-3.png_59-3_reflection_1.jpg", cv2.IMREAD_UNCHANGED)
ax = Image.open( "59-3.png_59-3_reflection_1.jpg" )

#img = cv2.imread("033_input_reflection_0.jpg", cv2.IMREAD_UNCHANGED)
#img[0,0,0] = 2


img = np.array(Image.open( "59-3.png_59-3_reflection_1.jpg" ))



#bw.mean()
gray = ax.convert('L')
# Let numpy do the heavy lifting for converting pixels to pure black or white
bw = np.asarray(gray).copy()
plt.imshow(bw)


#
#
#print(0.25*bw.max())
#print(0.25*img.max())
#print(img.mean())
#print(bw.mean())


print((img.mean() + 0.25*bw.max())/2)


Thresh = (img.mean() + 0.30*bw.max())/2




# Pixel range is 0...255, 256/2 = 128

#bw[bw < 0.25*img.max()] = 0
#bw[bw >= 0.25*img.max()] = 255 # White 


bw[bw < Thresh] = 0
bw[bw >= Thresh] = 255 # White  


plt.imshow(bw)

#bw[bw < 0.5*img.max()] = 0    # Black
#bw[bw > 0.9*img.max()] = 0    # Black
#
#
#
## Now we put it back in Pillow/PIL land
imfile = Image.fromarray(bw)

plt.figure()
plt.imshow(imfile)
#
#
imfile.save("test_mask.png")
#cv2.imwrite("alaki"+'.png',imfile)

#
#
#cv2.imwrite("asghar"+'.jpg',asghar)     





