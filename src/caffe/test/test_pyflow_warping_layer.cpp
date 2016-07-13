#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/python_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include <boost/python.hpp>


namespace caffe {

template <typename Dtype>
shared_ptr<Layer<Dtype> > XGetPythonLayer(const LayerParameter& param) {
    Py_Initialize();
    try {
        bp::object module = bp::import(param.python_param().module().c_str());
        bp::object layer = module.attr(param.python_param().layer().c_str())(param);
        return bp::extract<shared_ptr<PythonLayer<Dtype> > >(layer)();
    } catch (bp::error_already_set) {
        PyErr_Print();
        throw;
    }
}

template <typename TypeParam>
class FlowWarpingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  FlowWarpingLayerTest()
      : blob_im_(new Blob<Dtype>(2, 10, 1000, 1000)),
	blob_disp_(new Blob<Dtype>(2, 2, 1000, 1000)),
        blob_top_(new Blob<Dtype>()) {
    
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_im_);
    filler.Fill(this->blob_disp_);
    blob_bottom_vec_.push_back(this->blob_im_);
    blob_bottom_vec_.push_back(this->blob_disp_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~FlowWarpingLayerTest() { delete this->blob_im_; delete this->blob_disp_; delete this->blob_top_; }
  Blob<Dtype>* const blob_im_;
  Blob<Dtype>* const blob_disp_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types< CPUDevice<float> > TestDtypesOnGpu;
			 
TYPED_TEST_CASE(FlowWarpingLayerTest, TestDtypesOnGpu);

TYPED_TEST(FlowWarpingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PythonParameter* python_param = layer_param.mutable_python_param();
  python_param->set_module("flow_warping_layer");
  python_param->set_layer("FlowWarpingLayer");
  shared_ptr<Layer<Dtype> > layer = XGetPythonLayer<Dtype>(layer_param);
  //PythonLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}

}  // namespace caffe
