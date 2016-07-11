// --------------------------------------------------------
// Multitask Network Cascade
// Written by Haozhi Qi
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include <cfloat>

#include "caffe/flow_warp_layers.hpp"

namespace caffe {

template <typename Dtype>
void FlowWarpingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  vector<int> ones_size(1, channels_);
  ones_blob_.Reshape(ones_size);
  
  Dtype* ones = ones_blob_.mutable_cpu_data();
  
  for(int i = 0; i < channels_; i++) {
  	ones[i] = (Dtype) 1.0;
  }
}

template <typename Dtype>
void FlowWarpingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(channels_, bottom[0]->channels());
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  vector<int> size_nz;
  size_nz.push_back(bottom[0]->num());
  size_nz.push_back(width_ * height_ * 4);
  
  vector<int> size_ptrs;
  size_ptrs.push_back(bottom[0]->num());
  size_ptrs.push_back(width_ * height_ + 1);
  
  interp_coefs_blob_.Reshape(size_nz);
  partial_i_blob_.Reshape(size_nz);
  partial_j_blob_.Reshape(size_nz);
  indices_blob_.Reshape(size_nz);
  ptrs_blob_.Reshape(size_ptrs);
  
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void FlowWarpingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void FlowWarpingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(FlowWarpingLayer);
#endif

INSTANTIATE_CLASS(FlowWarpingLayer);
REGISTER_LAYER_CLASS(FlowWarping);

}  // namespace caffe
