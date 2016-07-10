#!/usr/bin/env sh

/home/amir/caffe/build/tools/caffe train \
    --solver=solver_davis_fcn8.prototxt \
    --snapshot= \
    -gpu 0 \
    2>&1 | tee resume_davis_fcn8"log$(date +'%m_%d_%y')".log
