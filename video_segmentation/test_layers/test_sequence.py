import caffe

import numpy as np
import os
from matplotlib.pyplot import imshow, show, figure
from skimage import io



def show(net, forward = False, score_threshold = 0, out = './result/', prefix = ''):
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
    
    if prefix != '':
	prefix = prefix + '_'
    for i in xrange(batch_size):
	name = out + '%s%d_cur_im.png' % (prefix, i)
	io.imsave(name, (cur_im[i].transpose((1,2,0))*255).astype('uint8'))
	name = out + '%s%d_masked_im.png' % (prefix, i)
	io.imsave(name, (masked_im[i].transpose((1,2,0))*255).astype('uint8'))
	name = out + '%s%d_next_im.png' % (prefix, i)
	io.imsave(name, (next_im[i].transpose((1,2,0))*255).astype('uint8'))
	name = out + '%s%d_label.png' % (prefix, i)
	io.imsave(name, (label[i][0]*255).astype('uint8'))
	name = out + '%s%d_score.png' % (prefix, i)
	io.imsave(name, (score[i]*255).astype('uint8'))
    






model = './train_davis_fcn8.prototxt'
#weights = './snapshots/davis_sequence_simple_iter_4163.caffemodel'
#weights = './snapshots/davis_sequence_simple_iter_2500.caffemodel'
weights = './fcn8_net.caffemodel'
# init
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(model, weights, caffe.TRAIN)

for i in range(1327):
    print 'Iteration', i
    show(net, True, 0, prefix = str(i))
