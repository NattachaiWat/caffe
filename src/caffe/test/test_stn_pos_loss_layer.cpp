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
#include "caffe/layers/stn_pos_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template < typename TypeParam>
class StnPosLossLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

    protected:
        StnPosLossLayerTest()
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
      virtual ~StnPosLossLayerTest()
        {
            delete blob_bottom_data_;
            delete blob_top_loss_;
        }      

        Blob<Dtype>* const blob_bottom_data_;
        Blob<Dtype>* const blob_top_loss_;
        vector<Blob<Dtype>*> blob_bottom_vec_;
        vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(StnPosLossLayerTest, TestDtypesAndDevices);

TYPED_TEST( StnPosLossLayerTest, TestForward){
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    StnPosLossParameter *stn_pos_loss_param = layer_param.mutable_stn_pos_loss_param();
    float threshold = 0.5;
    stn_pos_loss_param->set_threshold(threshold);
    vector<int> poss;
    poss.push_back(0);
    poss.push_back(2);
    stn_pos_loss_param->add_position(poss[0]);
    stn_pos_loss_param->add_position(poss[1]);
    
    StnPosLossLayer<Dtype> layer(layer_param);
    cout << "lidangwei: setup" << endl;
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    cout << "lidangwei: forward" << endl;
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    cout << "lidangwei: forward ok" << endl;
    // manually compute to compare
    // const Dtype margin = layer_param.stn_pos_loss_param().margin();
    const int num = this->blob_bottom_data_->num();
    const int channels = this->blob_bottom_data_->channels();
    const Dtype* data = this->blob_bottom_data_->mutable_cpu_data();
    Dtype loss(0.0);
    for (int i = 0; i < num; i++)
    {
        Dtype mdist = Dtype(0);
        for (int j=0; j<poss.size(); j++)
        {
            mdist = -data[i*channels+poss[j]] + threshold;
            if (mdist > 0)
            {
                loss += mdist;
            }
        }
    } 
    loss /= static_cast<Dtype>(num);
    EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6); 
}

TYPED_TEST(StnPosLossLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    StnPosLossParameter *stn_pos_loss_param = layer_param.mutable_stn_pos_loss_param();
    float threshold = 0.5;
    stn_pos_loss_param->set_threshold(threshold);
    vector<int> poss;
    poss.push_back(1);
    poss.push_back(3);
    stn_pos_loss_param->add_position(poss[0]);
    stn_pos_loss_param->add_position(poss[1]);
    StnPosLossLayer<Dtype> layer(layer_param);

    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_ );
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    // check gradient for first two layer
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);

}

} // end namespace caffe


