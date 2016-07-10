import caffe
import numpy as np
from matplotlib.pyplot import imshow, show, figure
from skimage import io
from skimage.transform import resize
from scipy.sparse import csr_matrix

path = 'test_warping_layer.prototxt'
img_path = '/home/amir/datasets/vsb/General_train_fullres/Images/lion/image028.jpg'
img  = io.imread(img_path)
img_resized = resize(img, (10,10))
data = img_resized[np.newaxis,:,:,:]
data = data.transpose((0,3,1,2))#np.zeros((1,3,10,10))
disp = np.ones((1,2,10,10))

caffe.set_mode_gpu()
net = caffe.Net(path, caffe.TRAIN)

print 'Data Shape:', net.blobs['data'].data.shape, 'Disp Shape:', net.blobs['disp'].data.shape
print 'Output shape:', net.blobs['data_i'].data.shape, net.blobs['data_j'].data.shape, net.blobs['sign_i'].data.shape, net.blobs['sign_i'].data.shape, net.blobs['indices'].data.shape, net.blobs['ptrs'].data.shape

net.blobs['data'].reshape(*data.shape)
net.blobs['data'].data[...] = data

net.blobs['disp'].reshape(*disp.shape)
net.blobs['disp'].data[...] = disp
net.forward()

data_i  = net.blobs['data_i'].data
data_j  = net.blobs['data_j'].data
sign_i  = net.blobs['sign_i'].data
sign_j  = net.blobs['sign_j'].data
indices = net.blobs['indices'].data.astype(np.int32)
ptrs = net.blobs['ptrs'].data.astype(np.int32)

bilinear_mat = csr_matrix(((data_i[0].ravel() * data_j[0].ravel()), indices[0].ravel(), ptrs[0].ravel()), shape = (100, 100))