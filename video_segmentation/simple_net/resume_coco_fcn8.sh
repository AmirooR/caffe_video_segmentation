#!/usr/bin/env sh

/home/amir/caffe/build/tools/caffe train \
    --solver=solver_coco_fcn8.prototxt \
    --snapshot=./snapshots/coco_fcn8_iter_17219.solverstate\
    -gpu 0 \
    2>&1 | tee resume_coco_fcn8"log$(date +'%m_%d_%y')".log
