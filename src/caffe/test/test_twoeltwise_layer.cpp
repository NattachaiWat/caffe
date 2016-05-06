#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/twoeltwise_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class TwoEltwiseLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TwoEltwiseLayerTest()
      : blob_bottom_a_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_b_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~TwoEltwiseLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_b_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TwoEltwiseLayerTest, TestDtypesAndDevices);

TYPED_TEST(TwoEltwiseLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TwoEltwiseParameter* twoeltwise_param = layer_param.mutable_twoeltwise_param();
  twoeltwise_param->set_operation(TwoEltwiseParameter_TwoEltwiseOp_PROD);
  shared_ptr<TwoEltwiseLayer<Dtype> > layer(
      new TwoEltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(TwoEltwiseLayerTest, TestProd) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TwoEltwiseParameter* twoeltwise_param = layer_param.mutable_twoeltwise_param();
  twoeltwise_param->set_operation(TwoEltwiseParameter_TwoEltwiseOp_PROD);
  shared_ptr<TwoEltwiseLayer<Dtype> > layer(
      new TwoEltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[i], 1e-4);
  }
}
TYPED_TEST(TwoEltwiseLayerTest, TestAbs) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TwoEltwiseParameter* twoeltwise_param = layer_param.mutable_twoeltwise_param();
  twoeltwise_param->set_operation(TwoEltwiseParameter_TwoEltwiseOp_PROD);
  twoeltwise_param->set_absout(true);
  shared_ptr<TwoEltwiseLayer<Dtype> > layer(
      new TwoEltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  Dtype absvalue = Dtype(0.0);
  for (int i = 0; i < count; ++i) {
    absvalue = in_data_a[i] * in_data_b[i];
    absvalue = absvalue > 0? absvalue : -1*absvalue;
    EXPECT_NEAR(data[i], absvalue, 1e-4);
  }
}

TYPED_TEST(TwoEltwiseLayerTest, TestSum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TwoEltwiseParameter* twoeltwise_param = layer_param.mutable_twoeltwise_param();
  twoeltwise_param->set_operation(TwoEltwiseParameter_TwoEltwiseOp_SUM);
  shared_ptr<TwoEltwiseLayer<Dtype> > layer(
      new TwoEltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] + in_data_b[i], 1e-4);
  }
}

TYPED_TEST(TwoEltwiseLayerTest, TestMax) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TwoEltwiseParameter* twoeltwise_param = layer_param.mutable_twoeltwise_param();
  twoeltwise_param->set_operation(TwoEltwiseParameter_TwoEltwiseOp_MAX);
  shared_ptr<TwoEltwiseLayer<Dtype> > layer(
      new TwoEltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(data[i],
              std::max(in_data_a[i], in_data_b[i]));
  }
}

TYPED_TEST(TwoEltwiseLayerTest, TestNumsqsum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TwoEltwiseParameter* twoeltwise_param = layer_param.mutable_twoeltwise_param();
  twoeltwise_param->set_operation(TwoEltwiseParameter_TwoEltwiseOp_PROD);
  twoeltwise_param->set_absout(false);
  twoeltwise_param->set_numsqsum(true);
  shared_ptr<TwoEltwiseLayer<Dtype> > layer(
      new TwoEltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_bottom_a_->count();
  const int num = this->blob_bottom_a_->num();
  const int chw = count/num;
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  std::cout<< count << " " << num << " " << chw << std::endl;
  Dtype absvalue = Dtype(0.0);
  for (int i = 0; i < num; ++i) {
    absvalue = 0;
    for(int j=0; j<chw; j++)
    {
        absvalue += (in_data_a[i*chw + j] * in_data_b[i*chw + j])*(in_data_a[i*chw + j] * in_data_b[i*chw + j]);
    }
    EXPECT_NEAR(data[i], absvalue, 1e-4);
  }
}
}  // namespace caffe
