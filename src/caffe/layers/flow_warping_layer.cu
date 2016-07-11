#include <cfloat>
#include "caffe/flow_warp_layers.hpp"
#include "caffe/util/tracker_math.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FlowWarpingFillCoefs(const int nthreads, Dtype* data_i, Dtype*
        data_j, Dtype* sign_i, Dtype* sign_j, int* indices, int* ptrs, int w, int h,
        const Dtype* disp) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // (i, j) is an element in the output
        int j = index % w;
        int i = index / w;

        //Find coeff. of o(i,j)
        //1) find r, c. disp is a nx2xhxw matrix
        Dtype y = disp[index] + i;
        Dtype x = disp[w * h + index] + j;
        int r = (int) y;
        int c = (int) x;

        //[r,c]
        //[r, c+1]
        //[r+1, c]
        //[r+1, c+1]
        //compute knn
        int nn_1 = r * w + c;
        int nn_2 = nn_1 + 1;
        int nn_3 = nn_1 + w;
        int nn_4 = nn_3 + 1;
        int indx = index * 4;
        ptrs[index] = indx;
        if(index == h * w - 1) {
            ptrs[h * w] = indx + 4;
        }
        
        if(0 <= r && r < h && 0 <= c && c < w) {
            data_i[indx] = 1 - (y - r);
            data_j[indx] = 1 - (x - c);
            sign_i[indx] = -1;
            sign_j[indx] = -1;
            indices[indx] = nn_1;

        }
        indx++;
        
        if(0 <= r && r < h && 0 <= c+1 && c+1 < w) {
            data_i[indx] = 1 - (y - r);
            data_j[indx] = x - c;
            sign_i[indx] = -1;
            sign_j[indx] = 1;
            indices[indx] = nn_2;
        }
        
        indx++;
        
        if(0 <= r+1 && r+1 < h && 0 <= c && c < w) {
            data_i[indx] = y - r;
            data_j[indx] = 1 - (x - c);
            sign_i[indx] = 1;
            sign_j[indx] = -1;
            indices[indx] = nn_3;
        }
        indx++;
        
        if(0 <= r+1 && r+1 < h && 0 <= c+1 && c+1 < w) {
            data_i[indx] = y - r;
            data_j[indx] = x - c;
            sign_i[indx] = 1;
            sign_j[indx] = 1;
            indices[indx] = nn_4;
        }
    }
}


template <typename Dtype>
void FlowWarpingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
					   const vector<Blob<Dtype>*>& top) {
  const Dtype* input_im = bottom[0]->gpu_data();
  const Dtype* disp = bottom[1]->gpu_data();
  Dtype* output_im = top[0]->mutable_gpu_data();


  Dtype* data_i = data_i_.mutable_gpu_data();
  Dtype* data_j = data_j_.mutable_gpu_data();
  Dtype* sign_i = sign_i_.mutable_gpu_data();
  Dtype* sign_j = sign_j_.mutable_gpu_data();
  Dtype* data_ij = data_ij_.mutable_gpu_data();
  int* indices = indices_blob_.mutable_gpu_data();
  int* ptrs = ptrs_blob_.mutable_gpu_data();
  
  const int count = width_ * height_;
  caffe_gpu_set(data_i_.count(), (Dtype) 0.0, data_i);
  caffe_gpu_set(data_j_.count(), (Dtype) 0.0, data_j);
  caffe_gpu_set(sign_i_.count(), (Dtype) 0.0, sign_i);
  caffe_gpu_set(sign_j_.count(), (Dtype) 0.0, sign_j);
  caffe_gpu_set(data_ij_.count(), (Dtype) 0.0, data_ij);
  caffe_gpu_set(indices_blob_.count(), (int) 0, indices);
  caffe_gpu_set(ptrs_blob_.count(), (int) 0, ptrs);

  for(int i = 0; i < top[0]->num(); i++) {

      // NOLINT_NEXT_LINE(whitespace/operators)
      FlowWarpingFillCoefs<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(count, data_i, data_j, sign_i, sign_j,
                  indices, ptrs, width_, height_, disp);
      CUDA_POST_KERNEL_CHECK;
     
      caffe_gpu_mul(count * 4, data_i, data_j, data_ij);
      //C = (A * B')' = B * A'
      tracker_gpu_csr_gemm_cusparse(CblasNoTrans, CblasTrans, count, channels_,
              count, (Dtype) 1.0 , count * 4, data_ij, indices, ptrs,
              input_im, (Dtype) 0.0, output_im, CblasColMajor);
      disp += count * 2;
      input_im += count * channels_;
      output_im += count * channels_;
      data_i += count * 4;
      data_j += count * 4;
      sign_i += count * 4;
      sign_j += count * 4;
      data_ij += count * 4;
      indices += count * 4;
      ptrs += count + 1;
  }
  //tracker_gpu_toDtype(count * 4, indices_blob.gpu_data(), indices_dtype);
  //tracker_gpu_toDtype(count + 1, ptrs_blob.gpu_data(), ptrs_dtype);
}

template <typename Dtype>
void FlowWarpingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
					    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int count = bottom[0]->count();
  
}
  
INSTANTIATE_LAYER_GPU_FUNCS(FlowWarpingLayer);
  
}  // namespace caffe

