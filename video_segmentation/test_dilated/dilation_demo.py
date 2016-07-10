# -*- coding: utf-8 -*-
"""
This package contains code for the "CRF-RNN" semantic image segmentation method, published in the 
ICCV 2015 paper Conditional Random Fields as Recurrent Neural Networks. Our software is built on 
top of the Caffe deep learning library.
 
Contact:
Shuai Zheng (szheng@robots.ox.ac.uk), Sadeep Jayasumana (sadeep@robots.ox.ac.uk), Bernardino Romera-Paredes (bernard@robots.ox.ac.uk)

Supervisor: 
Philip Torr (philip.torr@eng.ox.ac.uk)

For more information about CRF-RNN, please vist the project website http://crfasrnn.torr.vision.
"""
import os
import cPickle
import logging
import numpy as np
import pandas as pd
from PIL import Image as PILImage
#import Image
import cStringIO as StringIO
import caffe
import matplotlib.pyplot as plt


MODEL_FILE = 'dilation8_pascal_voc_deploy.prototxt'
PRETRAINED = 'dilation8_pascal_voc.caffemodel'
IMAGE_FILE = 'girl.bmp'
caffe.set_mode_gpu()
mean_vec = np.array([123.68, 116.779, 103.939], dtype=np.float32)



reshaped_mean_vec = mean_vec.reshape(1, 1, 3);
pallete = [0,0,0,
            128,0,0,
            0,128,0,
            128,128,0,
            0,0,128,
            128,0,128,
            0,128,128,
            128,128,128,
            64,0,0,
            192,0,0,
            64,128,0,
            192,128,0,
            64,0,128,
            192,0,128,
            64,128,128,
            192,128,128,
            0,64,0,
            128,64,0,
            0,192,0,
            128,192,0,
            0,64,128,
            128,64,128,
            0,192,128,
            128,192,128,
            64,64,0,
            192,64,0,
            64,192,0,
            192,192,0]


net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)


input_image = 255 * caffe.io.load_image(IMAGE_FILE)

     
label_margin = 186
width = input_image.shape[0]
height = input_image.shape[1]
maxDim = max(width,height)
scale = (900.0 - 2 * label_margin) / maxDim

im = PILImage.fromarray(np.uint8(input_image))
im = im.resize( [int(scale * s) for s in im.size] )
im = np.array(im)

# Subtract mean
im = im - reshaped_mean_vec

# Pad as necessary
cur_h, cur_w, cur_c = im.shape
im = np.pad(im, pad_width=(((900-cur_h)/2, (900-cur_h)/2), ((900-cur_w)/2, (900-cur_w)/2), (0, 0)), mode = 'constant', constant_values = 0)

im_t = im.transpose((2,0,1))
im_r = im_t.reshape(1,3,900,900)

net.blobs['data'].reshape(1,3, 900, 900)
net.blobs['data'].data[...] = im_r

net.forward()
# Get predictions
ct_final = net.blobs['ct_final'].data.squeeze()
fc_final = net.blobs['fc-final'].data.squeeze()


ct_mask = ct_final.argmax(axis=0).astype(np.uint8)
fc_mask = fc_final.argmax(axis=0).astype(np.uint8)


ct_mask_im = PILImage.fromarray(ct_mask)
ct_mask_im.putpalette(pallete)

plt.imshow(ct_mask_im)
plt.savefig('ct_mask.png')


fc_mask_im = PILImage.fromarray(fc_mask)
fc_mask_im.putpalette(pallete)

plt.figure()
plt.imshow(fc_mask)
plt.savefig('fc_mask.png')


