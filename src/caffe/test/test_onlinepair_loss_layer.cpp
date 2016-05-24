#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/other_functions.hpp"
#include "caffe/layers/onlinepair_loss_layer.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OnlinePairLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OnlinePairLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(20, 3, 1, 1)),
        blob_bottom_y_(new Blob<Dtype>(20, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    diff_ = new Blob<Dtype>(1,3,1,1);
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_y_->count(); ++i) {
      blob_bottom_y_->mutable_cpu_data()[i] = i%10; 
    }
    blob_bottom_vec_.push_back(blob_bottom_y_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~OnlinePairLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_y_;
    delete blob_top_loss_;
    delete diff_;
  }

  vector<PairDist> pairdist_pos_;
  vector<PairDist> pairdist_neg_;
  Blob<Dtype>* diff_; // the data malloc
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_y_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(OnlinePairLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(OnlinePairLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // manually compute to compare
  OnlinePairLossParameter *onlinepair_loss_param = 
    layer_param.mutable_onlinepair_loss_param();
  const Dtype margin_neg = 1.0;
  const Dtype margin_pos = 0.5;
  const Dtype hards_pos = 10;
  const Dtype hards_neg = 10;
  bool legacy_version = false; // check both true and false
  onlinepair_loss_param->set_legacy_version( legacy_version );
  onlinepair_loss_param->set_hards_pos( hards_pos );
  onlinepair_loss_param->set_hards_neg( hards_neg );
  onlinepair_loss_param->set_margin_neg( margin_neg );
  onlinepair_loss_param->set_margin_pos( margin_pos );
  OnlinePairLossLayer<Dtype> layer(layer_param);
  std::cout<< "doing foward ing "<<std::endl;
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  std::cout<< "forward ok now!" << std::endl;

  const int num = this->blob_bottom_data_->num();
  const int channels = this->blob_bottom_data_->channels();
  PairDist tmp;
  this->pairdist_pos_.clear();
  this->pairdist_neg_.clear();
  for(int i = 0; i < num - 1; i++)
    for( int j = i+1; j < num; j++)
    {
        caffe_sub(
            channels,
            this->blob_bottom_data_->cpu_data() + i*channels,
            this->blob_bottom_data_->cpu_data() + j*channels,
            this->diff_->mutable_cpu_data());
        tmp.dist = caffe_cpu_dot(channels, this->diff_->cpu_data(), this->diff_->cpu_data());
        tmp.first = i;
        tmp.second = j;
        tmp.flag = 
            this->blob_bottom_y_->mutable_cpu_data()[i] == this->blob_bottom_y_->mutable_cpu_data()[j] ? 1 : 0;
        if( tmp.flag == 1 )
        { this->pairdist_pos_.push_back(tmp); }
        else
        { this->pairdist_neg_.push_back(tmp); }
    }
  // std::sort(this->pairdist_pos_.begin(), this->pairdist_pos_.end(), pair_descend);
  // std::sort(this->pairdist_neg_.begin(), this->pairdist_neg_.end(), pair_ascend);
  if ( this->pairdist_pos_.size() > 0)
  {
    std::qsort(&this->pairdist_pos_[0], this->pairdist_pos_.size(), sizeof(PairDist), pair_descend_qsort);
  }
  if ( this->pairdist_neg_.size() > 0 )
  {
    std::qsort(&this->pairdist_neg_[0], this->pairdist_neg_.size(), sizeof(PairDist), pair_ascend_qsort);
  }
  // get the number of hard pos and hard neg counts
  int pos_num = this->pairdist_pos_.size() > hards_pos ? hards_pos : this->pairdist_pos_.size();
  int neg_num = this->pairdist_neg_.size() > hards_neg ? hards_neg : this->pairdist_neg_.size();
  std::cout << "pos_num: " << pos_num << std::endl;
  std::cout << "neg_num: " << neg_num << std::endl;
  std::cout << "hards_pos_num: " << hards_pos << std::endl;
  std::cout << "hards_neg_num: " << hards_neg << std::endl;
  std::cout << "pairdist_pos_.size: " << this->pairdist_pos_.size()  << std::endl;
  std::cout << "pairdist_neg_.size " << this->pairdist_neg_.size() << std::endl;

  Dtype loss(0);
  for (int i = 0; i < pos_num; i++)
  {
    if (legacy_version)
    {
        loss += std::max( Dtype(0.0), this->pairdist_pos_[i].dist - margin_pos);
    }
    else
    {
        Dtype dist = std::max( Dtype(0.0), sqrt( this->pairdist_pos_[i].dist)-margin_pos );
        loss += dist * dist;
    }
    // loss += this->pairdist_pos_[i].dist;
  }
  for (int j = 0; j < neg_num; j++)
  {
    if(legacy_version)
        loss += std::max( Dtype(0), margin_neg - this->pairdist_neg_[j].dist);
    else
    {
        Dtype dist = std::max(Dtype(0), margin_neg - sqrt( this->pairdist_neg_[j].dist) );
        loss += dist*dist;
    }
  }
  loss = loss / Dtype(pos_num + neg_num) / Dtype(2);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
}

TYPED_TEST(OnlinePairLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  OnlinePairLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the first bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
