#!/usr/bin/env sh

/home/amir/caffe/build/tools/caffe time -model train_coco_fcn8.prototxt -weights ./fcn8_net.caffemodel -gpu 0 -iterations 10
