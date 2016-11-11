#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/stn_pos_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void StnPosLossForwardGPU(const int N, const int cnt, const int L_pos, const Dtype threshold, 
    const Dtype* data, const int* pos, Dtype* loss_array)
{
    CUDA_KERNEL_LOOP(i, N) {
        Dtype mdist(0.0);
        for(int j=0; j<L_pos; j++)
        {
            mdist += threshold - data[i*cnt + pos[j]] > Dtype(0) ? threshold - data[i*cnt + pos[j]]: 0;
        }
        loss_array[i] = mdist;
    }
}

template <typename Dtype>
void StnPosLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    Dtype* loss_array = loss_.mutable_gpu_data();
    const Dtype* data = bottom[0]->mutable_gpu_data();
    const int* pos = pos_.gpu_data();
    int L_pos = pos_.count();
    int cnt = bottom[0]->count()/bottom[0]->num();
    
    StnPosLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, cnt, L_pos, 
        threshold, data, pos, loss_array);
    CUDA_POST_KERNEL_CHECK;

    Dtype loss = Dtype(0);
    caffe_gpu_asum(N, loss_.gpu_data(), &loss);
    loss /= N;
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void StnPosLossBackwardGPU(const int N, const int cnt, const int L_pos, const Dtype threshold,
    const Dtype* data, const int* pos, Dtype* diff)
{
    CUDA_KERNEL_LOOP(i, N) {
        Dtype mdist = Dtype(0);
        for(int j=0; j<L_pos; j++)
        {
            mdist = threshold - data[i*cnt + pos[j]] > Dtype(0) ? threshold - data[i*cnt + pos[j]] : 0;
            if (mdist > 0)
            {
                diff[i*cnt + pos[j]] = -1;
            }
        }
    }
}
template <typename Dtype>
void StnPosLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    const Dtype* data = bottom[0]->mutable_gpu_data();
    Dtype* diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(), Dtype(0.), diff);
    const int* pos = pos_.gpu_data();
    int L_pos = pos_.count();
    int cnt = bottom[0]->count()/bottom[0]->num();
    
    StnPosLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, cnt, L_pos,
        threshold, data, pos, diff);
    CUDA_POST_KERNEL_CHECK;

    caffe_gpu_scale(bottom[0]->count(), top[0]->cpu_diff()[0]/N, diff, diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(StnPosLossLayer);
}
