#include <algorithm>
#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/global_structure_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class GlobalStructureLossLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
    
  protected:
    GlobalStructureLossLayerTest()
        :blob_bottom_data_(new Blob<Dtype>(64, 32, 1, 1)),
        blob_bottom_y_(new Blob<Dtype>(64, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>())
    {
        // fille the values
        FillerParameter filler_param;
        filler_param.set_min(-10.0);
        filler_param.set_max(10.0);
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_data_);
        blob_bottom_vec_.push_back(blob_bottom_data_);
        // create the man-make value 
        /*
        blob_bottom_data_->mutable_cpu_data()[0] = 1/sqrt(2);
        blob_bottom_data_->mutable_cpu_data()[1] = 1/sqrt(2);
        blob_bottom_data_->mutable_cpu_data()[2] = 1/sqrt(2);
        blob_bottom_data_->mutable_cpu_data()[3] = 1/sqrt(2);
        blob_bottom_data_->mutable_cpu_data()[4] = 1/sqrt(2);
        blob_bottom_data_->mutable_cpu_data()[5] = 1/sqrt(2);
        blob_bottom_data_->mutable_cpu_data()[6] = 1/sqrt(2);
        blob_bottom_data_->mutable_cpu_data()[7] = 1/sqrt(2);
        */
        for (int i = 0; i < blob_bottom_y_->count(); ++i)
        {
            blob_bottom_y_->mutable_cpu_data()[i] = i % 16;
        }
        // normalize the input data 
        Dtype* data = blob_bottom_data_->mutable_cpu_data();
        int num = blob_bottom_data_->num();
        int channels = blob_bottom_data_->channels();
        for(int i=0; i<num; i++)
        {
            Dtype tmp = 0;
            for(int j=0; j<channels; j++)
            {
                tmp += data[i*channels+j]*data[i*channels+j];
            }
            tmp = sqrt(tmp);
            for(int j=0; j<channels; j++)
            {
                data[i*channels+j] /= tmp;
            }
        }
        blob_bottom_vec_.push_back(blob_bottom_y_);
        blob_top_vec_.push_back(blob_top_loss_);
    }
    virtual ~GlobalStructureLossLayerTest()
    {
        delete blob_bottom_data_;
        delete blob_bottom_y_;
        delete blob_top_loss_; 
    }
  
  // define some blobs
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_y_;
  Blob<Dtype>* const blob_top_loss_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

//TYPED_TEST_CASE(GlobalStructureLossLayerTest, TestDtypesAndDevices);
TYPED_TEST_CASE(GlobalStructureLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(GlobalStructureLossLayerTest, TestForward)
{
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    GlobalStructureLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // manually compute to compare
    const Dtype margin = layer_param.global_structure_loss_param().margin();
    const Dtype weight = layer_param.global_structure_loss_param().weight();
    const int num = this->blob_bottom_data_->num();
    const int channels = this->blob_bottom_data_->channels();
    // record the labels
    int C = 0;
    int D = channels;
    float label = 0;
    float index = 0;
    map<float, vector<float> > class_label_;
    for(int i=0; i<num; i++)
    {
        label = this->blob_bottom_y_->mutable_cpu_data()[i];
        if(class_label_.count(label))
        {
            class_label_[label][1] += 1;
        }
        else
        {
            vector<float> tmp;
            tmp.push_back(index);
            tmp.push_back(1);
            class_label_[label] = tmp;
            index += 1;
        }
    }
    C = int(index);
    Dtype* class_centers_ = new Dtype [C*D];
    // Dtype* center_matrix_ = new Dtype [num*D];
    // Dtype* delta_flag_ = new Dtype [C*C];
    Dtype* diff_centers_centers_ = new Dtype [C*C*D];
    memset(class_centers_, 0, sizeof(Dtype)*C*D);
    // memset(center_matrix_, 0, sizeof(Dtype)*num*D);
    // memset(delta_flag_, 0, sizeof(Dtype)*C*C);
    memset(diff_centers_centers_, 0, sizeof(Dtype)*C*C*D);
    Dtype* data = this->blob_bottom_data_->mutable_cpu_data();
    Dtype* y = this->blob_bottom_y_->mutable_cpu_data();
    // init the varibals
    for(int i=0; i<num; i++)
    {   
        // the class_centers_
        int offset = int(class_label_[y[i]][0]) * D;
        for(int d=0; d<D; d++)
        {
            class_centers_[offset + d]  += data[i*D + d];
        }
    }
    Dtype* center_l2_norm_ = new Dtype [C];
    for(map<float, vector<float> >::iterator it = class_label_.begin(); it != class_label_.end(); it++)
    {
        int offset = int(it->second[0])*D;
        for(int d=0; d<D; d++)
        {
            class_centers_[offset + d] /= it->second[1]; // normalize the vector
        }
        Dtype tmp = 0;
        for(int d=0; d<D; d++)
        {
            tmp += class_centers_[offset+d] * class_centers_[offset+d];
        }
        center_l2_norm_[int(it->second[0])] = sqrt(tmp);
        for(int d=0; d<D; d++)
        {
            class_centers_[offset + d] /= center_l2_norm_[int(it->second[0])];
        }
    }
    // init the diff_centers_centers_
    for(int c1 = 0; c1<C; c1++)
    {
        int offset1 = c1*C*D;
        for(int c2 = 0; c2<C; c2++)
        {
            int offset2 = c2*D;
            for(int d = 0; d<D; d++)
            {
                diff_centers_centers_[offset1 + offset2 + d] = 
                    class_centers_[c1*D + d] - class_centers_[c2*D + d];
            }
        }
    }
    // compute the intra loss
    Dtype loss(0);
    Dtype loss_inter(0);
    Dtype loss_intra(0);
    label = 0;
    Dtype* pt1 = NULL;
    Dtype* pt2 = NULL;
    for(int i=0; i<num; i++)
    {
        Dtype tmp = 0;
        label = y[i];
        pt1 = &data[i*D];
        pt2 = &class_centers_[int(class_label_[label][0])*D];
        for (int d=0; d<D; d++) { tmp += (pt1[d] - pt2[d])*(pt1[d] - pt2[d]); }
        loss_intra += tmp/class_label_[label][1];
    }
    loss_intra = loss_intra/2/C;
    // compute the inter loss
    for(int c1=0; c1<C-1; c1++)
    {   
        for(int c2=c1+1; c2<C; c2++)
        {
            Dtype tmp = 0;
            if (c1 == c2)
            {
                continue;
            }
            else
            {
                pt1 = &class_centers_[c1*D];
                pt2 = &class_centers_[c2*D];
                for(int d=0; d<D; d++)
                {
                    tmp += (pt1[d]-pt2[d])*(pt1[d] - pt2[d]);
                }
            }
            tmp = std::max(Dtype(0), margin - tmp);
            loss_inter += tmp;
        }
    }
    if (C > 1)
        loss_inter = loss_inter*2/C/(C-1);
    else
        loss_inter = 0;
    // compute the totally loss
    loss = loss_intra + weight*loss_inter;
    EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-2);
    // EXPECT_NEAR(this->blob_top_loss_->cpu_data()[1], loss_intra, 1e-2);
    // EXPECT_NEAR(this->blob_top_loss_->cpu_data()[2], loss_inter, 1e-2);
}

TYPED_TEST(GlobalStructureLossLayerTest, TestBackward)
{
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    GlobalStructureLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    // check the gradient for the first bottom layer
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
}



} // namespace std
