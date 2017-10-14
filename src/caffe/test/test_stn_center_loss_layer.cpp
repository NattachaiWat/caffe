/*

this file is reference to tangwei

*/
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/stn_center_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template < typename TypeParam>
class StnCenterLossLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

    protected:
        StnCenterLossLayerTest()
            : blob_bottom_data_( new Blob<Dtype>(100, 4, 1, 1)),
              // blob_bottom_y_( new Blob<Dtype>(512, 1, 1, 1)),
              blob_top_loss_( new Blob<Dtype>()) {
            // fill the values
            FillerParameter filler_param;
            filler_param.set_min(-2.0);
            filler_param.set_max(2.0);
            UniformFiller<Dtype> filler( filler_param );
            filler.Fill( this->blob_bottom_data_ );
            blob_bottom_vec_.push_back( blob_bottom_data_ );
            blob_top_vec_.push_back( blob_top_loss_ );
        }
      virtual ~StnCenterLossLayerTest()
        {
            delete blob_bottom_data_;
            delete blob_top_loss_;
        }      

        Blob<Dtype>* const blob_bottom_data_;
        Blob<Dtype>* const blob_top_loss_;
        vector<Blob<Dtype>*> blob_bottom_vec_;
        vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(StnCenterLossLayerTest, TestDtypesAndDevices);

TYPED_TEST( StnCenterLossLayerTest, TestForward){
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    StnCenterLossParameter *stn_center_loss_param = layer_param.mutable_stn_center_loss_param();
    float threshold = 0.5;
    float rate_hw = 2.0;
    stn_center_loss_param->set_threshold(threshold);
    stn_center_loss_param->set_rate_hw(rate_hw);
    vector<Dtype> centers;
    centers.push_back(Dtype(0.5));
    centers.push_back(Dtype(0.5));
    vector<int> poss;
    poss.push_back(1);
    poss.push_back(3);
    stn_center_loss_param->add_theta_bias(centers[0]);
    stn_center_loss_param->add_theta_bias(centers[1]);
    stn_center_loss_param->add_pos(poss[0]);
    stn_center_loss_param->add_pos(poss[1]);
    
    StnCenterLossLayer<Dtype> layer(layer_param);
    cout << "lidangwei: setup" << endl;
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    cout << "lidangwei: forward" << endl;
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    cout << "lidangwei: forward ok" << endl;
    // manually compute to compare
    // const Dtype margin = layer_param.stn_center_loss_param().margin();
    const int num = this->blob_bottom_data_->num();
    const int channels = this->blob_bottom_data_->channels();
    const Dtype* data = this->blob_bottom_data_->mutable_cpu_data();
    Dtype loss(0.0);
    for (int i = 0; i < num; i++)
    {
        Dtype mdist = Dtype(0);
        mdist = (data[i*channels+poss[0]]-centers[0])*(data[i*channels+poss[0]]-centers[0]); 
        mdist += rate_hw*rate_hw*(data[i*channels+poss[1]]-centers[1])*(data[i*channels+poss[1]]-centers[1]); 
        mdist -= threshold;
        if (mdist > 0)
        {
            loss += mdist/2;
        }
    } 
    loss /= static_cast<Dtype>(num);
    EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6); 
}

TYPED_TEST(StnCenterLossLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    StnCenterLossParameter *stn_center_loss_param = layer_param.mutable_stn_center_loss_param();
    float threshold = 0.5;
    stn_center_loss_param->set_threshold(threshold);
    vector<Dtype> centers;
    centers.push_back(Dtype(0.0));
    centers.push_back(Dtype(0.0));
    vector<int> poss;
    poss.push_back(1);
    poss.push_back(3);
    stn_center_loss_param->add_theta_bias(centers[0]);
    stn_center_loss_param->add_theta_bias(centers[1]);
    stn_center_loss_param->add_pos(poss[0]);
    stn_center_loss_param->add_pos(poss[1]);
    StnCenterLossLayer<Dtype> layer(layer_param);

    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_ );
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    // check gradient for first two layer
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);

}

} // end namespace caffe


