import caffe
import numpy as np
from matplotlib.pyplot import imshow, show, figure
from skimage import io
from skimage.transform import resize
from scipy.sparse import csr_matrix
import struct

coef_path = 'coefs_0.txt'
indices_path = 'indices_0.txt'
ptrs_path = 'ptrs_0.txt'

coef_file = open(coef_path, 'rb');
indices_file = open(indices_path, 'rb');
ptrs_file = open(ptrs_path, 'rb');


count = struct.unpack('i', coef_file.read(4))[0];
ptrs_count = struct.unpack('i', ptrs_file.read(4))[0];

coef_file.read(4);
indices_file.read(8);
ptrs_file.read(4);


coefs = struct.unpack('f' * count, coef_file.read(4 * count))
indices = struct.unpack('i' * count, indices_file.read(4 * count))
ptrs = struct.unpack('i' * ptrs_count, ptrs_file.read(4 * ptrs_count))

coefs = np.array(coefs)
indices = np.array(indices, dtype=np.int)
ptrs = np.array(ptrs, dtype=np.int)


mat = csr_matrix((coefs, indices, ptrs))