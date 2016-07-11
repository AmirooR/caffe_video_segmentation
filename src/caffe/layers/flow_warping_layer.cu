#include <cfloat>
#include "caffe/flow_warp_layers.hpp"
#include "caffe/util/tracker_math.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

__device__ void is_free(const int i, const int* nn_ids, int* free_indices, int& cur_index) {
    if((nn_ids[0] != i) && (nn_ids[1] != i) && (nn_ids[2] != i) && (nn_ids[3] != i)) {
	free_indices[cur_index] = i;
	cur_index++;
    }
}

// 12, 0, 13, 1
// 0, 1, 2, 3
// 1, 0, 3, 2 -- 1, 0, 3, 2 

// 2, 0,  3, 1
// 1,3,0,2
__device__ void sort4(const int* in_num, int* inverse_map) {
    int forward_map[4] = {0,1,2,3};
    
    if(in_num[0] > in_num[1]) {
        inverse_map[0] = 1;
        inverse_map[1] = 0;
        forward_map[0] = 1;
        forward_map[1] = 0;
    }

    if(in_num[2] > in_num[3]) {
        inverse_map[2] = 3;
        inverse_map[3] = 2;
        forward_map[2] = 3;
        forward_map[3] = 2;
    }

    if(in_num[forward_map[0]] > in_num[forward_map[2]]) {
        int tmp = forward_map[0];
        forward_map[0] = forward_map[2];
        forward_map[2] = tmp;
        
    }

    if(in_num[forward_map[1]] > in_num[forward_map[3]]) {
        int tmp = forward_map[1];
        forward_map[1] = forward_map[3];
        forward_map[3] = tmp;
   
    }

    if(in_num[forward_map[1]] > in_num[forward_map[2]]) {
        int tmp = forward_map[1];
        forward_map[1] = forward_map[2];
        forward_map[2] = tmp;
   
    }
    inverse_map[forward_map[0]] = 0;
    inverse_map[forward_map[1]] = 1;
    inverse_map[forward_map[2]] = 2;
    inverse_map[forward_map[3]] = 3;
}
__device__ void find_nns(const int r, const int c, int* nn_ids, int* inverse_map, bool* is_valid, int w, int h) { 
    
    is_valid[0] = 0 <= r && r < h && 0 <= c && c < w;
    is_valid[1] = 0 <= r && r < h && 0 <= c+1 && c+1 < w;
    is_valid[2] = 0 <= r+1 && r+1 < h && 0 <= c && c < w;
    is_valid[3] = 0 <= r+1 && r+1 < h && 0 <= c+1 && c+1 < w;

    nn_ids[0] = r * w + c;
    nn_ids[1] = nn_ids[0] + 1;
    nn_ids[2] = nn_ids[0] + w;
    nn_ids[3] = nn_ids[2] + 1;
    
    //Is that necessary
    //if(is_valid[0] && is_valid[1] && is_valid[2] && is_valid[3]) {
	//nn_order[0] = 0;
	//nn_order[1] = 1;
	//nn_order[2] = 2;
	//nn_order[3] = 3;
	//return;
    //}
    //For nns that do not have a valid location, we still should assign a unique id to be able to make
    //the sparse matrix. As long as the correspoing data is set to zero the result won't change.
    //We choose among id = {0,1,2,3} whichever are not taken.
    int free_indx[4];
    int cur_index = 0;
    is_free(0, nn_ids, free_indx, cur_index);
    is_free(1, nn_ids, free_indx, cur_index);
    is_free(2, nn_ids, free_indx, cur_index);
    is_free(3, nn_ids, free_indx, cur_index);
    
    //Assigning free sptos
    cur_index = 0;
    if(!is_valid[0]) {
	nn_ids[0] = free_indx[cur_index];
	cur_index++;
    }
	
    if(!is_valid[1]) {
	nn_ids[1] = free_indx[cur_index];
	cur_index++;
    }
    
    if(!is_valid[2]) {
	nn_ids[2] = free_indx[cur_index];
	cur_index++;
    }
    
    if(!is_valid[3]) {
	nn_ids[3] = free_indx[cur_index];
	cur_index++;
    }
    
    //Choose the order so that nn_ids[nn_order[.]] is sorted
    sort4(nn_ids, inverse_map);
} 

template <typename Dtype>
__global__ void FlowWarpingBilinearCoefs(const int nthreads, Dtype* bi_coefs, Dtype* partial_i, Dtype*
        partial_j, int* indices, int* ptrs, int w, int h,
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
        Dtype rem_x = y - r;
        Dtype rem_y = x - c;
        Dtype rem_1x = 1 - rem_x;
        Dtype rem_1y = 1 - rem_y;


        int data_offset = index * 4;
        ptrs[index] = data_offset;
        if(index == h * w - 1) {
            ptrs[h * w] = data_offset + 4;
        }

        //[r,c]
        //[r, c+1]
        //[r+1, c]
        //[r+1, c+1]
        //compute knn
        int nn_ids[4];
        int inv_map[4] = {0,1,2,3};
        bool is_valid[4];
        find_nns(r, c, nn_ids, inv_map, is_valid, w, h);
        // 12, 0, 13, 1
        // 1, 3, 0, 2
        // 2, 0,  3, 1
        int data_indx = data_offset + inv_map[0];
        indices[data_indx] = nn_ids[0];
        if(is_valid[0]) {
            bi_coefs[data_indx] = rem_1y * rem_1x;
            partial_i[data_indx] = -rem_1x;
            partial_j[data_indx] = -rem_1y;
        }

        data_indx = data_offset + inv_map[1];
        indices[data_indx] = nn_ids[1];
        if(is_valid[1]) {
            bi_coefs[data_indx] = rem_1y * rem_x;
            partial_i[data_indx] = -rem_x;
            partial_j[data_indx] = rem_1y;
        }

        data_indx = data_offset + inv_map[2];
        indices[data_indx] = nn_ids[2];
        if(is_valid[2]) {
            bi_coefs[data_indx] = rem_y * rem_1x;
            partial_i[data_indx] = rem_1x;
            partial_j[data_indx] = -rem_y;
        }

        data_indx = data_offset + inv_map[3];
        indices[data_indx] = nn_ids[3];
        if(is_valid[3]) {
            bi_coefs[data_indx] = rem_y * rem_x;
            partial_i[data_indx] = rem_x;
            partial_j[data_indx] = rem_y;
        }
    }
}


template <typename Dtype>
void FlowWarpingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
					   const vector<Blob<Dtype>*>& top) {
  const Dtype* input_im = bottom[0]->gpu_data();
  const Dtype* disp = bottom[1]->gpu_data();
  Dtype* output_im = top[0]->mutable_gpu_data();
  
  //LOG(ERROR) << "IN FORWARD";

  Dtype* partial_i = partial_i_blob_.mutable_gpu_data();
  Dtype* partial_j = partial_j_blob_.mutable_gpu_data();
  Dtype* interp_coefs = interp_coefs_blob_.mutable_gpu_data();
  int* indices = indices_blob_.mutable_gpu_data();
  int* ptrs = ptrs_blob_.mutable_gpu_data();
  const int count = width_ * height_;
  caffe_gpu_set(partial_i_blob_.count(), (Dtype) 0.0, partial_i);
  caffe_gpu_set(partial_j_blob_.count(), (Dtype) 0.0, partial_j);
  caffe_gpu_set(interp_coefs_blob_.count(), (Dtype) 0.0, interp_coefs);
  caffe_gpu_set(indices_blob_.count(), (int) -1, indices);
  caffe_gpu_set(ptrs_blob_.count(), (int) 0, ptrs);

  for(int i = 0; i < top[0]->num(); i++) {
     
      // NOLINT_NEXT_LINE(whitespace/operators)
      FlowWarpingBilinearCoefs<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(count, interp_coefs, partial_i, partial_j,
                  indices, ptrs, width_, height_, disp);
      CUDA_POST_KERNEL_CHECK;

      //out = in * A^t = (A * in^t)^t
      tracker_gpu_csr_gemm_cusparse(CblasNoTrans, CblasTrans, count, channels_,
              count, (Dtype) 1.0, count * 4, interp_coefs, indices, ptrs,
              input_im, (Dtype) 0.0, output_im, CblasColMajor);
      
      disp += count * 2;
      input_im += count * channels_;
      output_im += count * channels_;
      partial_i += count * 4;
      partial_j += count * 4;
      interp_coefs += count * 4;
      indices += count * 4;
      ptrs += count + 1;
  }
  
/*  const Dtype* cpu_interp_coefs = interp_coefs_blob_.cpu_data();
  const int* cpu_indices = indices_blob_.cpu_data();
  const int* cpu_ptrs = ptrs_blob_.cpu_data();
  for(int i = 0; i < top[0]->num(); i++) {
      tracker_saveMat(std::string("coefs_") + char('0' + i) + std::string(".txt"), cpu_interp_coefs, count * 4, count * 4);
      tracker_saveMat(std::string("indices_") + char('0' + i) + std::string(".txt"), cpu_indices, count * 4, count * 4);
      tracker_saveMat(std::string("ptrs_") + char('0' + i) + std::string(".txt"), cpu_ptrs, count + 1, count + 1);
      
      cpu_interp_coefs += count * 4;
      cpu_indices += count * 4;
      cpu_ptrs += count + 1;
  }*/
  
}

template <typename Dtype>
void FlowWarpingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
					    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
LOG(ERROR) << "IN BACKWARD";
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* im_diff = bottom[0]->mutable_gpu_diff();
  Dtype* disp_diff = bottom[1]->mutable_gpu_diff();
  
  
  const Dtype* input_im = bottom[0]->gpu_data();
  const Dtype* ones = ones_blob_.gpu_data();
  const Dtype* partial_i = partial_i_blob_.gpu_data();
  const Dtype* partial_j = partial_j_blob_.gpu_data();
  const Dtype* interp_coefs = interp_coefs_blob_.gpu_data();
  const int* indices = indices_blob_.gpu_data();
  const int* ptrs = ptrs_blob_.gpu_data();
  const int count = width_ * height_;
  for(int i = 0; i < top[0]->num(); i++) {
      if(propagate_down[1]) {
      	//Let z' = d out_{c,i,j}/d disp_{0, i,j}: 
      	//z' = input_im * partial_i^t = (partial_i * input_im^t)^t
      	//use im_diff as a temporary variable
      	tracker_gpu_csr_gemm_cusparse(CblasNoTrans, CblasTrans, count, channels_,
        		      count, (Dtype) 1.0 , count * 4, partial_i, indices, ptrs,
        		      input_im, (Dtype) 0.0, im_diff, CblasColMajor);
        //disp_{0}' = sum(z' .* top_diff, 0) 
        caffe_gpu_mul(count * channels_, im_diff, top_diff, im_diff);
        
        //sum(.,0)
        caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, count, channels_,
        		(Dtype) 1.0, im_diff, ones, (Dtype) 0.0, disp_diff);
        
       
        //The same steps for disp_{1}:
      	tracker_gpu_csr_gemm_cusparse(CblasNoTrans, CblasTrans, count, channels_,
        		      count, (Dtype) 1.0 , count * 4, partial_j, indices, ptrs,
        		      input_im, (Dtype) 0.0, im_diff, CblasColMajor);

        caffe_gpu_mul(count * channels_, im_diff, top_diff, im_diff);
        caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, count, channels_,
        		(Dtype) 1.0, im_diff, ones, (Dtype) 0.0, disp_diff + count);
        
      }      
      
      //out = in * A^t ==> in' = out' * A = (A^t * out'^t)^t
  	  tracker_gpu_csr_gemm_cusparse(CblasTrans, CblasTrans, count, channels_,
            count, (Dtype) 1.0 , count * 4, interp_coefs, indices, ptrs,
            top_diff, (Dtype) 0.0, im_diff,
            CblasColMajor);      
                
      top_diff += count * channels_;
      im_diff += count * channels_;
      disp_diff += count * 2;
      
      input_im += count * channels_;
      partial_i += count * 4;
      partial_j += count * 4;
      interp_coefs += count * 4;
      indices += count * 4;
      ptrs += count + 1;
  }
  


}
  
INSTANTIATE_LAYER_GPU_FUNCS(FlowWarpingLayer);
  
}  // namespace caffe

