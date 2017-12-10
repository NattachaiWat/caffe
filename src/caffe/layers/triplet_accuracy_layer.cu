/*
* triplet_loss_layer.cu
*
*/

#include <algorithm>
#include <vector>


#include "caffe/layers/triplet_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletAccuracyLayer<Dtype>::Forward_gpu(
   const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
   const int count = bottom[0]->count();
   caffe_gpu_sub(
     count,
     bottom[0]->gpu_data(),  // a
     bottom[1]->gpu_data(),  // p
     diff_ap_.mutable_gpu_data());  // a_i-p_i
   caffe_gpu_mul(
     count,
     diff_ap_.mutable_gpu_data(),  // a_i-p_i
     diff_ap_.mutable_gpu_data(),
     diff_ap_.mutable_gpu_data());  // (a_i-p_i)^2
   caffe_gpu_sub(
     count,
     bottom[0]->gpu_data(),  // a
     bottom[2]->gpu_data(),  // n
     diff_an_.mutable_gpu_data());  // a_i-n_i
   caffe_gpu_mul(
     count,
     diff_an_.mutable_gpu_data(),  // a_i-n_i
     diff_an_.mutable_gpu_data(),
     diff_an_.mutable_gpu_data());  // (a_i-n_i)^2
   caffe_gpu_gemv(
     CblasNoTrans,
     bottom[0]->num(),
     bottom[0]->channels(),
     Dtype(1.0),                                         //alpha
     diff_ap_.gpu_data(),  // (a_i-p_i)^2                // A
     summer_vec_channel_.gpu_data(),                             // x
     Dtype(0.0),                                         //belta
     dist_sq_ap_.mutable_gpu_data());  // \Sum (a_i-p_i)^2  //y
   caffe_gpu_gemv(
     CblasNoTrans,
     bottom[0]->num(),
     bottom[0]->channels(),
     Dtype(1.0),                                         //alpha
     diff_an_.gpu_data(),  // (a_i-n_i)^2                // A
     summer_vec_channel_.gpu_data(),                             // x
     Dtype(0.0),                                         //belta
     dist_sq_an_.mutable_gpu_data());  // \Sum (a_i-n_i)^2  //y

   Dtype margin = this->layer_param_.triplet_accuracy_param().margin();
   Dtype accuracy = Dtype(0);
   for (int i = 0; i < bottom[0]->num(); ++i) {
     // std::cout<< "gpu diff is :" << margin << " "<< dist_sq_ap_.cpu_data()[i]<<" " <<dist_sq_an_.cpu_data()[i] << std::endl;
     accuracy += margin + dist_sq_ap_.cpu_data()[i] - dist_sq_an_.cpu_data()[i] >= 0 ? 0 : 1;
   }
   top[0]->mutable_cpu_data()[0] = accuracy/bottom[0]->num();
 }

template <typename Dtype>
void TripletAccuracyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
}
INSTANTIATE_LAYER_GPU_FUNCS(TripletAccuracyLayer);

}  // namespace caffe
