import caffe

import numpy as np
import os
from matplotlib.pyplot import imshow, show, figure
from skimage import io


model = './val_davis_sequence_simple.prototxt'
#weights = './snapshots/davis_sequence_simple_iter_4177.caffemodel'
weights = './snapshots/davis_sequence_simple_iter_2500.caffemodel'
iters = 1376/4
# init
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(model, weights, caffe.TRAIN)
val_error = 0.0

for i in xrange(iters):
    if i != 0:
	print 'Validation Error at iteration', i, ':', val_error/i
    net.forward()
    val_error = val_error + net.blobs['loss'].data

print 'Validation Error:', val_error/iters
