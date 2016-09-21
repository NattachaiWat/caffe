#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/stn_box_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void StnBoxLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	LossLayer<Dtype>::LayerSetUp(bottom, top);

	string prefix = "\t\tBox Loss Layer:: LayerSetUp: \t";

	threshold = (Dtype) this->layer_param_.stn_box_loss_param().threshold();
	std::cout<<prefix<<"Getting threshold value = "<<threshold<<std::endl;
    
	CHECK(threshold > 0) << "Error: threshold should be larger than zero.";
}

template <typename Dtype>
void StnBoxLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	N = bottom[0]->num();

	vector<int> loss_shape(1);
	loss_shape[0] = N;
	loss_.Reshape(loss_shape);

    top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void StnBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	/* not implemented */
	// CHECK(false) << "Error: not implemented.";
    // use the simple hing-loss to do this task
    Dtype* loss_array = loss_.mutable_cpu_data();
    Dtype* data = bottom[0]->mutable_cpu_data();
    for(int i=0; i<N; i++)
    {
        Dtype mdist = Dtype(0);
        // process the four types of loss
        mdist += std::max(Dtype(0), (data[i*4+1]-data[i*4])*(data[i*4+1]-data[i*4]) - threshold);
        mdist += std::max(Dtype(0), (data[i*4+1]+data[i*4])*(data[i*4+1]+data[i*4]) - threshold);
        mdist += std::max(Dtype(0), (data[i*4+3]-data[i*4+2])*(data[i*4+3]-data[i*4+2]) - threshold);
        mdist += std::max(Dtype(0), (data[i*4+3]+data[i*4+2])*(data[i*4+3]+data[i*4+2]) - threshold);
        loss_array[i] = mdist/2;
    }
    Dtype loss = Dtype(0);
    loss = caffe_cpu_asum(N, loss_array);
    loss /= N;
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void StnBoxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	/* not implemented */
	// CHECK(false) << "Error: not implemented.";
    const Dtype* data = bottom[0]->mutable_cpu_data();
    Dtype* diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), diff);
    for(int i=0; i<N; i++)
    {
        Dtype mdist = Dtype(0);
        mdist = std::max(Dtype(0), (data[i*4+1]-data[i*4])*(data[i*4+1]-data[i*4]) - threshold);
        if (mdist > 0)
        {
            diff[i*4+1] += data[i*4+1]-data[i*4];
            diff[i*4] += -1*(data[i*4+1]-data[i*4]);
        }
        mdist = std::max(Dtype(0), (data[i*4+1]+data[i*4])*(data[i*4+1]+data[i*4]) - threshold);
        if (mdist > 0)
        {
            diff[i*4+1] += data[i*4+1]+data[i*4];
            diff[i*4] += data[i*4+1]+data[i*4];
        }
        mdist = std::max(Dtype(0), (data[i*4+3]-data[i*4+2])*(data[i*4+3]-data[i*4+2]) - threshold);
        if (mdist > 0)
        {
            diff[i*4+3] += data[i*4+3]-data[i*4+2];
            diff[i*4+2] += -1*(data[i*4+3]-data[i*4+2]);
        }
        mdist = std::max(Dtype(0), (data[i*4+3]+data[i*4+2])*(data[i*4+3]+data[i*4+2]) - threshold);
        if (mdist > 0)
        {
            diff[i*4+3] += data[i*4+3]+data[i*4+2];
            diff[i*4+2] += data[i*4+3]+data[i*4+2];
        }
    }
    caffe_cpu_scale(bottom[0]->count(), top[0]->cpu_diff()[0]/N, diff, diff);
}

#ifdef CPU_ONLY
STUB_GPU(StnBoxLayer);
#endif

INSTANTIATE_CLASS(StnBoxLossLayer);
REGISTER_LAYER_CLASS(StnBoxLoss);

}  // namespace caffe
