$CAFFE_PATH train \
    -solver=$SOLVER \
    -weights=$WEIGHTS \
    -gpu 0 \
    2>&1 | tee ./log/$LOG_NAME"log$$(date +'%m_%d_%y')".log
