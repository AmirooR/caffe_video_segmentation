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

template <>
void tracker_gpu_csr_gemm_cusparse<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, int nzz, const float* A, const int* indices, const int* ptr, const float* B, const float beta,
    float* C, const CBLAS_ORDER orderC) {

  //std::cout << "M: " << M << " N: " << N << " K: " << K << " NZZ: " << nzz <<"\n"  ;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cusparseOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

  float* A_t;
  int* A_t_indices;
  int* A_t_ptr;
  
  int msparse = (TransA == CblasNoTrans) ? M : K;
  int ksparse = (TransA == CblasNoTrans) ? K : M;
  
  bool reuiqre_transpose_A = (cuTransA == CUSPARSE_OPERATION_TRANSPOSE) && (cuTransB == CUSPARSE_OPERATION_TRANSPOSE);
  //LOG(ERROR) << "Require Transpose A? " << reuiqre_transpose_A;
  if (reuiqre_transpose_A){
    CUDA_CHECK(cudaMalloc((void**)&A_t, sizeof(float)*nzz));
    CUDA_CHECK(cudaMalloc((int**)&A_t_indices, sizeof(int)*nzz));
    CUDA_CHECK(cudaMalloc((int**)&A_t_ptr, sizeof(int)*(ksparse+1)));
    CUSPARSE_CHECK(cusparseScsr2csc(Caffe::cusparse_handle(), msparse, ksparse, nzz, A, ptr, indices, A_t, A_t_indices, A_t_ptr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));
  }
  
  if (orderC == CblasRowMajor){
    float* Ct;
    CUDA_CHECK(cudaMalloc((void**)&Ct, sizeof(float)*M*N));
    const float zero = 0.0;
    const float one = 1.0;
    if (reuiqre_transpose_A){
      CUSPARSE_CHECK(cusparseScsrmm2(Caffe::cusparse_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE, cuTransB, ksparse, N, msparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A_t, A_t_ptr, A_t_indices, B,  ldb, &zero, Ct, M));
      CUDA_CHECK(cudaFree(A_t));
      CUDA_CHECK(cudaFree(A_t_indices));
      CUDA_CHECK(cudaFree(A_t_ptr));
    }else{
      CUSPARSE_CHECK(cusparseScsrmm2(Caffe::cusparse_handle(), cuTransA, cuTransB, msparse, N, ksparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A, ptr, indices, B,  ldb, &zero, Ct, M));
    }
    CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), CUBLAS_OP_T , CUBLAS_OP_N, N, M, &one, Ct, M, &beta, C, N, C, N));
    CUDA_CHECK(cudaFree(Ct));
  }else{
      
    //this is the default of CUSPARSE by the Matrix B is by default rowmajor
    if (reuiqre_transpose_A){
      CUSPARSE_CHECK(cusparseScsrmm2(Caffe::cusparse_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE, cuTransB, ksparse, N, msparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A_t, A_t_ptr, A_t_indices, B,  ldb, &beta, C, M));      
      CUDA_CHECK(cudaFree(A_t));
      CUDA_CHECK(cudaFree(A_t_indices));
      CUDA_CHECK(cudaFree(A_t_ptr));
    }else{
      //LOG(ERROR) << "HERE!!!! " << (cuTransA == CUSPARSE_OPERATION_TRANSPOSE) << ", " << (cuTransB == CUSPARSE_OPERATION_TRANSPOSE) << ", " << msparse << ", " << N << ", " << ksparse << ", " << nzz << ", " << ldb << ", " << M;
      CUSPARSE_CHECK(cusparseScsrmm2(Caffe::cusparse_handle(), cuTransA, cuTransB, msparse, N, ksparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A, ptr, indices, B,  ldb, &beta, C, M));
    }
  }
}


template <>
void tracker_gpu_csr_gemm_cusparse<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, int nzz, const double* A, const int* indices, const int* ptr, const double* B, const double beta,
    double* C, const CBLAS_ORDER orderC) {

  //std::cout << "M: " << M << "N: " << N << "K: " << K << "NZZ: " << nzz  ;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cusparseOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

  double* A_t;
  int* A_t_indices;
  int* A_t_ptr;
  
  int msparse = (TransA == CblasNoTrans) ? M : K;
  int ksparse = (TransA == CblasNoTrans) ? K : M;
  
  bool reuiqre_transpose_A = (cuTransA == CUSPARSE_OPERATION_TRANSPOSE) && (cuTransB == CUSPARSE_OPERATION_TRANSPOSE);
  if (reuiqre_transpose_A){
    CUDA_CHECK(cudaMalloc((void**)&A_t, sizeof(double)*nzz));
    CUDA_CHECK(cudaMalloc((int**)&A_t_indices, sizeof(int)*nzz));
    CUDA_CHECK(cudaMalloc((int**)&A_t_ptr, sizeof(int)*(ksparse+1)));
    CUSPARSE_CHECK(cusparseDcsr2csc(Caffe::cusparse_handle(), msparse, ksparse, nzz, A, ptr, indices, A_t, A_t_indices, A_t_ptr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));
  }

  
  if (orderC == CblasRowMajor){
    double* Ct;
    CUDA_CHECK(cudaMalloc((void**)&Ct, sizeof(double)*M*N));
    const double zero = 0.0;
    const double one = 1.0;
    if (reuiqre_transpose_A){
      CUSPARSE_CHECK(cusparseDcsrmm2(Caffe::cusparse_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE, cuTransB, ksparse, N, msparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A_t, A_t_ptr, A_t_indices, B,  ldb, &zero, Ct, M));
      CUDA_CHECK(cudaFree(A_t));
      CUDA_CHECK(cudaFree(A_t_indices));
      CUDA_CHECK(cudaFree(A_t_ptr));
    }else{
      CUSPARSE_CHECK(cusparseDcsrmm2(Caffe::cusparse_handle(), cuTransA, cuTransB, msparse, N, ksparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A, ptr, indices, B,  ldb, &zero, Ct, M));
    }
    CUBLAS_CHECK(cublasDgeam(Caffe::cublas_handle(), CUBLAS_OP_T , CUBLAS_OP_N, N, M, &one, Ct, M, &beta, C, N, C, N));
    CUDA_CHECK(cudaFree(Ct));
  }else{
    //this is the default of CUSPARSE by the Matrix B is by default rowmajor
    if (reuiqre_transpose_A){
      CUSPARSE_CHECK(cusparseDcsrmm2(Caffe::cusparse_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE, cuTransB, ksparse, N, msparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A_t, A_t_ptr, A_t_indices, B,  ldb, &beta, C, M));      
      CUDA_CHECK(cudaFree(A_t));
      CUDA_CHECK(cudaFree(A_t_indices));
      CUDA_CHECK(cudaFree(A_t_ptr));
    }else{
      CUSPARSE_CHECK(cusparseDcsrmm2(Caffe::cusparse_handle(), cuTransA, cuTransB, msparse, N, ksparse,nzz, &alpha, Caffe::cusparse_mat_descr(), A, ptr, indices, B,  ldb, &beta, C, M));
    }
  }
}


}  // namespace caffe
