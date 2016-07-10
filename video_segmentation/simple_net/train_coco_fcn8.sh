/home/amir/caffe/build/tools/caffe train \
    -solver=solver_coco_fcn8.prototxt \
    -weights=./fcn8_net.caffemodel \
    -gpu 0 \
    2>&1 | tee train_coco_fcn8"log$(date +'%m_%d_%y')".log
