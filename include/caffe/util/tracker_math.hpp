#ifndef CAFFE_UTIL_TRACKER_MATH_H_
#define CAFFE_UTIL_TRACKER_MATH_H_

#include <stdint.h>

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"
extern "C" {
  #include <clapack.h>
}
 
namespace caffe {

template <typename Dtype>
void tracker_printMat(std::ostream& buffer, const Dtype* mat, int col, int count);

template <typename Dtype>
void tracker_saveMat(string filename, const Dtype* mat, int col, int count);

template <typename Dtype>
void tracker_gpu_toInt(int n, const Dtype* in, int* out);

template <typename Dtype>
void tracker_gpu_toDtype(int n, const int* in, Dtype* out);
}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
