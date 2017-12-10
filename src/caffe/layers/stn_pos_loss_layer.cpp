#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/stn_pos_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void StnPosLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	LossLayer<Dtype>::LayerSetUp(bottom, top);

	string prefix = "\t\tPos Loss Layer:: LayerSetUp: \t";

	threshold = (Dtype) this->layer_param_.stn_pos_loss_param().threshold();
	std::cout<<prefix<<"Getting threshold value = "<<threshold<<std::endl;
    
    // pos
    int n_pos = this->layer_param_.stn_pos_loss_param().position_size();
    int cnt = bottom[0]->count()/bottom[0]->num();
    pos_.Reshape(1,n_pos, 1, 1);
    int* pos = pos_.mutable_cpu_data();
    for(int i=0; i<n_pos; i++) 
    {
        pos[i] = this->layer_param_.stn_pos_loss_param().position(i);
    }
    if (n_pos == 0)
    {
        for(int i=0; i<cnt; i++) pos[i] = i;
    }
	CHECK(threshold > 0) << "Error: threshold should be larger than zero.";
    CHECK(cnt >= n_pos) << "the count of pos must less than the data dim.";
}

template <typename Dtype>
void StnPosLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	N = bottom[0]->num();

	vector<int> loss_shape(1);
	loss_shape[0] = N;
	loss_.Reshape(loss_shape);

    top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void StnPosLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	/* not implemented */
	// CHECK(false) << "Error: not implemented.";
    // use the simple hing-loss to do this task, max(Dtype(0), threshold-theta)
    int L_pos = pos_.count();
    int cnt = bottom[0]->count()/bottom[0]->num();

    Dtype* loss_array = loss_.mutable_cpu_data();
    const Dtype* data = bottom[0]->cpu_data();
    const int* pos = pos_.cpu_data();
    for(int i=0; i<N; i++)
    {
        loss_array[i] = 0;
        for(int j=0; j<L_pos; j++)
        {
            loss_array[i] += std::max(Dtype(0), threshold - data[i*cnt+pos[j]] );
        }
    }
    Dtype loss = caffe_cpu_asum(N, loss_array);
    loss /= N;
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void StnPosLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	/* not implemented */
	// CHECK(false) << "Error: not implemented.";
    int L_pos = pos_.count();
    int cnt = bottom[0]->count()/bottom[0]->num();

    const Dtype* data = bottom[0]->cpu_data();
    Dtype* diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), diff);
    const int* pos = pos_.cpu_data();
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<L_pos; j++)
        {
            Dtype mdist = std::max(Dtype(0), threshold - data[i*cnt+pos[j]] );
            if (mdist > 0)
            {
                diff[i*cnt+pos[j]] = -1;
            }
        }
    }
    // weight the diff
    caffe_cpu_scale(bottom[0]->count(), top[0]->cpu_diff()[0]/N, diff, diff);
}

#ifdef CPU_ONLY
STUB_GPU(StnPosLossLayer);
#endif

INSTANTIATE_CLASS(StnPosLossLayer);
REGISTER_LAYER_CLASS(StnPosLoss);

}  // namespace caffe
