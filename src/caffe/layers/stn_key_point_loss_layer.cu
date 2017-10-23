#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/stn_key_point_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void StnKeyPointLossForwardGPU(const int N, const int channels, const Dtype threshold, 
    const Dtype* data, const Dtype* gt_bias, const Dtype rate_hw, const int* gt_pos, Dtype* loss_array)
{
    CUDA_KERNEL_LOOP(i, N) {
        Dtype mdist(0.0);
        mdist = (data[i*channels + gt_pos[0]] - gt_bias[2*i+0])*(data[i*channels + gt_pos[0]] - gt_bias[2*i+0]);
        mdist += rate_hw*rate_hw*(data[i*channels + gt_pos[1]] - gt_bias[2*i+1])*(data[i*channels + gt_pos[1]] - gt_bias[2*i+1]);
        mdist = mdist - threshold;
        if (mdist > 0.0)
        {
            loss_array[i] = mdist/2;
        }
        else
        {
            loss_array[i] = 0.0;
        }
    }
}

template <typename Dtype>
void StnKeyPointLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    const Dtype* data = bottom[0]->gpu_data();
    Dtype* loss_array = loss_.mutable_gpu_data();
    caffe_gpu_set(loss_.count(), (Dtype)0, loss_array);

    const int channels = bottom[0]->channels();
    const Dtype* gt_bias = bottom[1]->gpu_data();
    const int* gt_pos = pos_.gpu_data();

    StnKeyPointLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, channels, threshold_,
        data, gt_bias, rate_hw_, gt_pos, loss_array);
    CUDA_POST_KERNEL_CHECK; 

    Dtype loss;
    caffe_gpu_asum(N_, loss_array, &loss);
    loss /= N_;
    
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void StnKeyPointLossBackwardGPU(const int N, const int channels, 
    const Dtype* data, const Dtype* gt_bias, const int* gt_pos, const Dtype rate_hw, const Dtype* loss, Dtype* diff)
{
    CUDA_KERNEL_LOOP(i, N) {
        if (loss[i] > 0 )
        {
            int index = i*channels + gt_pos[0];
            diff[ index ] = data[ index ] - gt_bias[2*i+0];
            index = i*channels + gt_pos[1];
            diff[ index ] = rate_hw*(data[ index ] - gt_bias[2*i+1]);
        }
    }
}
template <typename Dtype>
void StnKeyPointLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* theta = bottom[0]->gpu_data();
    const Dtype* loss_array = loss_.gpu_data();
    const int channels = bottom[0]->channels();
    const Dtype* gt_bias = bottom[1]->gpu_data(); 
    const int* gt_pos = pos_.gpu_data();
    StnKeyPointLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, channels, theta,
        gt_bias, gt_pos, rate_hw_, loss_array, bottom[0]->mutable_gpu_diff() ); 
    CUDA_POST_KERNEL_CHECK;

    caffe_gpu_scal(bottom[0]->count(), top[0]->cpu_diff()[0]/N_, bottom[0]->mutable_gpu_diff());
}
INSTANTIATE_LAYER_GPU_FUNCS(StnKeyPointLossLayer);
}
