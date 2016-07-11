import caffe
import numpy as np
from matplotlib.pyplot import imshow, show, figure
from skimage import io
from skimage.transform import resize
from scipy.sparse import csr_matrix

path = 'test_warping_layer.prototxt'
img_paths = ['input_0.jpg', 'input_1.jpg']
im_shape = (100, 100)

caffe.set_mode_gpu()
net = caffe.Net(path, caffe.TRAIN)

print 'Data Shape:', net.blobs['data'].data.shape, 'Disp Shape:', net.blobs['disp'].data.shape
print 'Output shape:', net.blobs['out'].data.shape

net.blobs['data'].reshape(len(img_paths), 3, *im_shape)
net.blobs['disp'].reshape(len(img_paths), 2, *im_shape)

for i, img_path in enumerate(img_paths):
    img  = io.imread(img_path)
    img_resized = resize(img, im_shape)
    io.imsave('resized_%d.png' % i, img_resized)
    data = img_resized.transpose((2,0,1))
    disp = 10*(i+1)*((-1)**i)*np.ones((2,)+im_shape)

    net.blobs['data'].data[i][...] = data
    net.blobs['disp'].data[i][...] = disp

net.forward()
out = net.blobs['out'].data

for i in xrange(len(img_paths)):
    img = out[i].transpose((1,2,0))
    io.imsave('out_%d.png' % i, img)
