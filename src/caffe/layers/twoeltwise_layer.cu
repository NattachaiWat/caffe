#include <cfloat>
#include <vector>

#include "caffe/layers/twoeltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (bottom_data_a[index] > bottom_data_b[index]) {
        top_data[index] = bottom_data_a[index];
    } else {
      top_data[index] = bottom_data_b[index]; 
    }
  }
}

template <typename Dtype>
void TwoEltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int chw = count/num;
  Dtype* top_data = preoutput_.mutable_gpu_data();
  Dtype* output_data = top[0]->mutable_gpu_data();
  
  switch (op_) {
  case TwoEltwiseParameter_TwoEltwiseOp_PROD:
    caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    break;
  case TwoEltwiseParameter_TwoEltwiseOp_SUM:
    caffe_gpu_add(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    break;
  case TwoEltwiseParameter_TwoEltwiseOp_MAX:
    MaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_data);
    break;
  case TwoEltwiseParameter_TwoEltwiseOp_SUB:
    caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
  if (this->layer_param_.twoeltwise_param().absout() == true )
  {
    caffe_gpu_abs(count, top_data, top_data);
  }
  if( this->layer_param_.twoeltwise_param().numsqsum() == true )
  {
    caffe_gpu_powx(count, preoutput_.mutable_gpu_data(), Dtype(2), preoutput_.mutable_gpu_data());
    std::cout << "li" << num << " " << chw << std::endl;
    caffe_gpu_gemv(CblasNoTrans, num, chw, Dtype(1.0), preoutput_.mutable_gpu_data(), 
            summer_vec_.mutable_gpu_data(), Dtype(0.0), top[0]->mutable_gpu_data());
  }else{
    caffe_copy(count, top_data, output_data);
  }
}

template <typename Dtype>
void TwoEltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(TwoEltwiseLayer);

}  // namespace caffe
