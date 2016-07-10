#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/tracker_math.hpp"
#include "cublas_v2.h"
#define THREADS_PER_BLOCK_CSR 32

namespace caffe {

template <typename Dtype>
__global__ void toInt_kernel(int n, const Dtype* in, int* out)
{
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = static_cast<int>(in[index]);
  }
}

template <typename Dtype>
void tracker_gpu_toInt(int n, const Dtype* in, int* out)
{
  toInt_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, in, out);
}

template void tracker_gpu_toInt<float>(int n, const float* in, int* out);
template void tracker_gpu_toInt<double>(int n, const double* in, int* out);



template <typename Dtype>
__global__ void toDtype_kernel(int n, const int* in, Dtype* out)
{
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = static_cast<Dtype>(in[index]);
  }
}

template <typename Dtype>
void tracker_gpu_toDtype(int n, const int* in, Dtype* out)
{
  toDtype_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, in, out);
}

template void tracker_gpu_toDtype<float>(int n, const int* in, float* out);
template void tracker_gpu_toDtype<double>(int n, const int* in, double* out);
}  // namespace caffe
