import caffe
import numpy as np
from matplotlib.pyplot import imshow, show, figure
from skimage import io
from skimage.transform import resize
from scipy.sparse import csr_matrix
from scipy.io import savemat
from util import write_flo_file
def forward(net):
    net.forward()
    return net.blobs['img1'].data, net.blobs['img2'].data, net.blobs['flow'].data

path = 'test_netflow_datalayer.prototxt'
caffe.set_mode_gpu()
net = caffe.Net(path, caffe.TRAIN)
print 'Output shape:', net.blobs['img1'].data.shape, net.blobs['img2'].data.shape, net.blobs['flow'].data.shape
mean=np.array([0.410602, 0.431021, 0.448553]).reshape(1,3,1,1)

for i in range(10):
    im1, im2, flow = forward(net)
    im1 += mean
    im2 += mean
    im1[im1 > 1] = 1;im1[im1 < 0] = 0;im2[im2 > 1] = 1;im2[im2 < 0] = 0
    batch_size = im1.shape[0]
    flow = flow[:, ::-1, :, :]
    flow = flow.transpose((0, 2, 3, 1))

    for j in range(batch_size):
        k = i * batch_size + j
        io.imsave('im1_%05d.png' % k, im1[j].transpose((1,2,0)))
        io.imsave('im2_%05d.png' % k, im2[j].transpose((1,2,0)))  
        write_flo_file('flo_%05d.flo'  % k, flow[j])
        #savemat('flo_%05d.mat'  % k, {'flow':flow[j]})
