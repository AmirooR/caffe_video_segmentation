// --------------------------------------------------------
// Multitask Network Cascade
// Written by Haozhi Qi
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include <cfloat>

#include "caffe/flow_warp_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void FlowWarpingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    //TODO: check roi_warp if parameters are needed
}

template <typename Dtype>
void FlowWarpingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  
  LOG(ERROR) << "IN RESHAPE: " << width_ << ", " << height_ << ", " << width_ * height_ * 4;
  
  //top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);
  
  vector<int> size;
  size.push_back(bottom[0]->num());
  size.push_back(width_ * height_ * 4);
  
  vector<int> size_ptrs;
  size_ptrs.push_back(bottom[0]->num());
  size_ptrs.push_back(width_ * height_ + 1);
  
  //data_i
  top[0]->Reshape(size);
  //data_j
  top[1]->Reshape(size);
  //sign_i
  top[2]->Reshape(size);
  //sign_j
  top[3]->Reshape(size);
  //indices
  top[4]->Reshape(size);
  //ptrs
  top[5]->Reshape(size_ptrs);




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
