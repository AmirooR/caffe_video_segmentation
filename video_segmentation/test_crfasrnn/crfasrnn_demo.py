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


MODEL_FILE = 'TVG_CRFRNN_COCO_VOC.prototxt'
PRETRAINED = 'TVG_CRFRNN_COCO_VOC.caffemodel'
IMAGE_FILE ='/home/amir/datasets/kitti/2011_09_26/2011_09_26_drive_0005_sync/image_03/data/0000000151.png'
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


net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TRAIN)


input_image = 255 * caffe.io.load_image(IMAGE_FILE)


width = input_image.shape[0]
height = input_image.shape[1]
maxDim = max(width,height)
scale = 500.0 / maxDim

im = PILImage.fromarray(np.uint8(input_image))
im = im.resize( [int(scale * s) for s in im.size] )
im = np.array(im)
#im = image[:,:,::-1]


# Subtract mean
im = im - reshaped_mean_vec

# Pad as necessary
cur_h, cur_w, cur_c = im.shape
pad_h = 500 - cur_h
pad_w = 500 - cur_w
im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)

im_t = im.transpose((2,0,1))
im_r = im_t.reshape(1,3,500,500)

net.blobs['data'].reshape(1,3, 500, 500)
net.blobs['data'].data[...] = im_r

net.forward()
# Get predictions
fc8_score = net.blobs['score'].data.squeeze()
crf_score = net.blobs['pred'].data.squeeze()


fc8_mask = fc8_score.argmax(axis=0).astype(np.uint8)
crf_mask = crf_score.argmax(axis=0).astype(np.uint8)


#fc8_mask = fc8_mask[0:cur_h, 0:cur_w]
fc8_mask_im = PILImage.fromarray(fc8_mask)
fc8_mask_im.putpalette(pallete)

plt.imshow(fc8_mask_im)
plt.savefig('fc8_mask.png')


crf_mask = crf_mask[0:cur_h, 0:cur_w]
crf_mask_im = PILImage.fromarray(crf_mask)
crf_mask_im.putpalette(pallete)

plt.figure()
plt.imshow(crf_mask)
plt.savefig('crf_mask.png')


