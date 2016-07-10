import caffe

import numpy as np
import os


weights = './simple_net.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)
solver.step(4000)
