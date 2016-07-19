// --------------------------------------------------------
// Multitask Network Cascade
// Modified from caffe-fast-rcnn (https://github.com/rbgirshick/caffe-fast-rcnn)
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#ifndef CAFFE_FAST_RCNN_LAYERS_HPP_
#define CAFFE_FAST_RCNN_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/power_layer.hpp"
namespace caffe {

template <typename Dtype>
class FlowWarpingLayer : public Layer<Dtype> {
 public:
  explicit FlowWarpingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FlowWarping"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_;
  int width_;

  Blob<Dtype> partial_i_blob_;
  Blob<Dtype> partial_j_blob_;
  Blob<Dtype> interp_coefs_blob_;
  Blob<int> indices_blob_;
  Blob<int> ptrs_blob_;
  Blob<Dtype> ones_blob_;
};

/**
 * @brief Computes the L1 or L2 Loss, optionally on the L2 norm along channels
 *
 */

//Forward declare
template <typename Dtype> class ConvolutionLayer;
template <typename Dtype> class EltwiseLayer;

template <typename Dtype>
class L1LossLayer : public LossLayer<Dtype> {
 public:
  explicit L1LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), sign_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);    
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "L1Loss"; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
  
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }

 protected:
  /// @copydoc L1LossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> sign_, mask_, plateau_l2_;
  float scale_;
  Dtype normalize_coeff_;
  
  // Extra layers to do the dirty work using already implemented stuff
  shared_ptr<EltwiseLayer<Dtype> > diff_layer_;
  Blob<Dtype> diff_;
  vector<Blob<Dtype>*> diff_top_vec_;
  shared_ptr<PowerLayer<Dtype> > square_layer_;
  Blob<Dtype> square_output_;
  vector<Blob<Dtype>*> square_top_vec_;
  shared_ptr<ConvolutionLayer<Dtype> > sum_layer_;
  Blob<Dtype> sum_output_;
  vector<Blob<Dtype>*> sum_top_vec_;
  shared_ptr<PowerLayer<Dtype> > sqrt_layer_;
  Blob<Dtype> sqrt_output_;
  vector<Blob<Dtype>*> sqrt_top_vec_;
};


/**
 * @brief Phil's Downsample Layer
 * Takes a blob and downsamples width and height to given size
 */
template <typename Dtype>
class DownsampleLayer : public Layer<Dtype> {
 public:
  explicit DownsampleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Downsample"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline bool AllowBackward() const { LOG(WARNING) << "DownsampleLayer does not do backward."; return false; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  
  int top_width_;
  int top_height_;
};

}  // namespace caffe

#endif  // CAFFE_FAST_RCNN_LAYERS_HPP_
