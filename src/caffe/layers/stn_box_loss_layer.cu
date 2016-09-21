#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/stn_box_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void StnBoxLossForwardGPU(const int N, const Dtype threshold, 
    const Dtype* data, Dtype* loss_array)
{
    CUDA_KERNEL_LOOP(i, N) {
        Dtype mdist = Dtype(0);
        Dtype temp = Dtype(0);
        temp = (data[i*4+1]-data[i*4])*(data[i*4+1]-data[i*4]) - threshold;
        mdist += temp > 0 ? temp : 0;
        temp = (data[i*4+1]+data[i*4])*(data[i*4+1]+data[i*4]) - threshold;
        mdist += temp > 0 ? temp : 0;
        temp = (data[i*4+3]-data[i*4+2])*(data[i*4+3]-data[i*4+2]) - threshold;
        mdist += temp > 0 ? temp : 0;
        temp = (data[i*4+3]+data[i*4+2])*(data[i*4+3]+data[i*4+2]) - threshold;
        mdist += temp > 0 ? temp : 0;
        loss_array[i] = mdist/2;
    }
}

template <typename Dtype>
void StnBoxLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    Dtype* data = bottom[0]->mutable_gpu_data();
    Dtype* loss_array = loss_.mutable_gpu_data();

    StnBoxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, threshold,
        data, loss_array);
    CUDA_POST_KERNEL_CHECK;

    Dtype loss;
    caffe_gpu_asum(N, loss_array, &loss);
    loss /= N;
    
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void StnBoxLossBackwardGPU(const int N, const Dtype threshold,  
    const Dtype* data, Dtype* diff)
{
    CUDA_KERNEL_LOOP(i, N) {
        Dtype mdist = Dtype(0);
        mdist = (data[i*4+1]-data[i*4])*(data[i*4+1]-data[i*4]) - threshold;
        if (mdist > 0)
        {
            diff[i*4+1] += data[i*4+1]-data[i*4];
            diff[i*4] += -1*(data[i*4+1]-data[i*4]);
        }
        mdist = (data[i*4+1]+data[i*4])*(data[i*4+1]+data[i*4]) - threshold;
        if (mdist > 0)
        {
            diff[i*4+1] += data[i*4+1]+data[i*4];
            diff[i*4] += data[i*4+1]+data[i*4];
        }
        mdist = (data[i*4+3]-data[i*4+2])*(data[i*4+3]-data[i*4+2]) - threshold;
        if (mdist > 0)
        {
            diff[i*4+3] += data[i*4+3]-data[i*4+2];
            diff[i*4+2] += -1*(data[i*4+3]-data[i*4+2]);
        }
        mdist = (data[i*4+3]+data[i*4+2])*(data[i*4+3]+data[i*4+2]) - threshold;
        if (mdist > 0)
        {
            diff[i*4+3] += data[i*4+3]+data[i*4+2];
            diff[i*4+2] += data[i*4+3]+data[i*4+2];
        }
    }
}
template <typename Dtype>
void StnBoxLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    Dtype* data = bottom[0]->mutable_gpu_data();
    Dtype* diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(), Dtype(0), diff);

    StnBoxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, threshold,
        data, diff); 
    CUDA_POST_KERNEL_CHECK;

    caffe_gpu_scale(bottom[0]->count(), top[0]->cpu_diff()[0]/N, diff, diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(StnBoxLossLayer);
}
