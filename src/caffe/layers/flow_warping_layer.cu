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
        
        if(0 <= nn_1 && nn_1 < w * h) {
            data_i[indx] = 1 - (y - r);
            data_j[indx] = 1 - (x - c);
            sign_i[indx] = -1;
            sign_j[indx] = -1;
            indices[indx] = nn_1;

        }
        indx++;
        
        if(0 <= nn_2 && nn_2 < w * h) {
            data_i[indx] = 1 - (y - r);
            data_j[indx] = x - c;
            sign_i[indx] = -1;
            sign_j[indx] = 1;
            indices[indx] = nn_2;
        }
        
        indx++;
        
        if(0 <= nn_3 && nn_3 < w * h) {
            data_i[indx] = y - r;
            data_j[indx] = 1 - (x - c);
            sign_i[indx] = 1;
            sign_j[indx] = -1;
            indices[indx] = nn_3;
        }
        indx++;
        
        if(0 <= nn_4 && nn_4 < w * h) {
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

  Dtype* data_i = top[0]->mutable_gpu_data();
  Dtype* data_j = top[1]->mutable_gpu_data();
  Dtype* sign_i = top[2]->mutable_gpu_data();
  Dtype* sign_j = top[3]->mutable_gpu_data();
  Dtype* indices_dtype = top[4]->mutable_gpu_data();
  Dtype* ptrs_dtype = top[5]->mutable_gpu_data();

  Blob<int> indices_blob;
  Blob<int> ptrs_blob;

  indices_blob.Reshape(top[4]->shape());
  ptrs_blob.Reshape(top[5]->shape());

  int* indices = indices_blob.mutable_gpu_data();
  int* ptrs = ptrs_blob.mutable_gpu_data();
  
  const int count = width_ * height_;
  caffe_gpu_set(count * 4, (Dtype) 0.0, data_i);
  caffe_gpu_set(count * 4, (Dtype) 0.0, data_j);
  caffe_gpu_set(count * 4, (Dtype) 0.0, sign_i);
  caffe_gpu_set(count * 4, (Dtype) 0.0, sign_j);
  caffe_gpu_set(count * 4, (int) 0, indices);
  caffe_gpu_set(count + 1, (int) 0, ptrs);

  for(int i = 0; i < top[0]->num(); i++) {

      // NOLINT_NEXT_LINE(whitespace/operators)
      FlowWarpingFillCoefs<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(count, data_i, data_j, sign_i, sign_j,
                  indices, ptrs, width_, height_, disp);

      CUDA_POST_KERNEL_CHECK;
      data_i += count * 4;
      data_j += count * 4;
      sign_i += count * 4;
      sign_j += count * 4;
      indices += count * 4;
      ptrs += count + 1;
  }
  tracker_gpu_toDtype(count * 4, indices_blob.gpu_data(), indices_dtype);
  tracker_gpu_toDtype(count + 1, ptrs_blob.gpu_data(), ptrs_dtype);
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

