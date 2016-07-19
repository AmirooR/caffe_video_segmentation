import caffe

import numpy as np
import os
from matplotlib.pyplot import imshow, show, figure
from skimage import io



def show(net, forward = False, score_threshold = 0, out = './result/'):
    if forward:
      net.forward()
    cur_im = net.blobs['cur_im'].data
    masked_im = net.blobs['masked_im'].data
    next_im = net.blobs['next_im'].data
    label = net.blobs['label'].data
    score = np.exp(net.blobs['score'].data)
    score = score[:,1]/score.sum(1)
    if score_threshold > 0:
      score = score > score_threshold
    batch_size = cur_im.shape[0]
    for i in xrange(batch_size):
	name = out + '%d_cur_im.png' % i
	io.imsave(name, (cur_im[i].transpose((1,2,0))*255).astype('uint8'))
	name = out + '%d_masked_im.png' % i
	io.imsave(name, (masked_im[i].transpose((1,2,0))*255).astype('uint8'))
	name = out + '%d_next_im.png' % i
	io.imsave(name, (next_im[i].transpose((1,2,0))*255).astype('uint8'))
	name = out + '%d_label.png' % i
	io.imsave(name, (label[i][0]*255).astype('uint8'))
	name = out + '%d_score.png' % i
	io.imsave(name, (score[i]*255).astype('uint8'))
    






model = './train_davis_sequence_simple.prototxt'
weights = './snapshots/davis_simple_iter_500.caffemodel'
#weights = './snapshots/coco_deepmask_iter_41666.caffemodel'
# init
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(model, weights, caffe.TRAIN)

