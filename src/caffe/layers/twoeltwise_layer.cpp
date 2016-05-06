#include <cfloat>
#include <vector>

#include "caffe/layers/twoeltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TwoEltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  op_ = this->layer_param_.twoeltwise_param().operation();
}

template <typename Dtype>
void TwoEltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  int num = bottom[0]->num();
  preoutput_.ReshapeLike(*bottom[0]);
  summer_vec_.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  for( int i=0; i< summer_vec_.count(); i++) 
  {
    summer_vec_.mutable_cpu_data()[i] = 1;
  }
  if (this->layer_param_.twoeltwise_param().numsqsum() == true)
  {  
    top[0]->Reshape(num, 1, 1, 1);
  }else {
    top[0]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void TwoEltwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data_a = NULL;
  const Dtype* bottom_data_b = NULL;
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int chw = count/num;
  Dtype* top_data = preoutput_.mutable_cpu_data();
  Dtype* output_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case TwoEltwiseParameter_TwoEltwiseOp_PROD:
    caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
    break;
  case TwoEltwiseParameter_TwoEltwiseOp_SUM:
    caffe_add(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
    break;
  case TwoEltwiseParameter_TwoEltwiseOp_MAX:
    bottom_data_a = bottom[0]->cpu_data();
    bottom_data_b = bottom[1]->cpu_data();
    for (int idx = 0; idx < count; ++idx) {
      if (bottom_data_a[idx] > bottom_data_b[idx]) {
        top_data[idx] = bottom_data_a[idx];  // maxval
      } else {
        top_data[idx] = bottom_data_b[idx];  // maxval
      }
    }
    break;
  case TwoEltwiseParameter_TwoEltwiseOp_SUB:
    caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
  if( this->layer_param_.twoeltwise_param().absout() == true )
  {
    caffe_abs(count, top_data, top_data);
  }
  if( this->layer_param_.twoeltwise_param().numsqsum() == true )
  {
    for( int i=0; i<num; i++)
    {
       output_data[i] = caffe_cpu_dot(chw, top_data + (i*chw), top_data + (i*chw)); 
    }
  }else{
    caffe_copy(count, top_data, output_data); 
  }
}

#ifdef CPU_ONLY
STUB_GPU(TwoEltwiseLayer);
#endif

INSTANTIATE_CLASS(TwoEltwiseLayer);
REGISTER_LAYER_CLASS(TwoEltwise);

}  // namespace caffe
