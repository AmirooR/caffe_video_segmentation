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

    
template<typename Dtype>
void tracker_gpu_csr_gemm_cusparse(const CBLAS_TRANSPOSE TransA,
                          const CBLAS_TRANSPOSE TransB, const int M, const int N,
                          const int K, const Dtype alpha, int nzz, const Dtype* A,
                          const int* indices, const int* ptr, const Dtype* B,
                          const Dtype beta, Dtype* C, const CBLAS_ORDER orderC);

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
