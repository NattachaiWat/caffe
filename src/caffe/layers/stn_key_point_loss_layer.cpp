#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/stn_key_point_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void StnKeyPointLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	LossLayer<Dtype>::LayerSetUp(bottom, top);

	string prefix = "\t\tKey Point Loss Layer:: LayerSetUp: \t";

	threshold_ = (Dtype) this->layer_param_.stn_key_point_loss_param().threshold();
	rate_hw_ = (Dtype) this->layer_param_.stn_key_point_loss_param().rate_hw();
	std::cout << prefix << "Getting threshold value = " << threshold_ << std::endl;
    
    // load the theta_bias value and pos, two values
    CHECK(this->layer_param_.stn_key_point_loss_param().pos_size() == 2 || 
        this->layer_param_.stn_key_point_loss_param().pos_size() == 0) << "the length of pos must be 0 or 2!";

    pos_.Reshape(1,2,1,1);
    // theta_bias_.Reshape(1,2,1,1);
    int* pos = pos_.mutable_cpu_data(); 
    // Dtype* theta_bias = theta_bias_.mutable_cpu_data();

    if (this->layer_param_.stn_key_point_loss_param().pos_size() == 0)
    {
        pos[0] = 0;
        pos[1] = 1;
    }
    else
    {
        pos[0] =  this->layer_param_.stn_key_point_loss_param().pos(0);
        pos[1] =  this->layer_param_.stn_key_point_loss_param().pos(1);
    }
	CHECK(threshold_ >= 0) << "Error: threshold should be larger than zero.";
}

template <typename Dtype>
void StnKeyPointLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	N_ = bottom[0]->num();
	vector<int> loss_shape(1);
	loss_shape[0] = N_;
	loss_.Reshape(loss_shape);
    top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void StnKeyPointLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	/* not implemented */
	// CHECK(false) << "Error: not implemented.";
    // use the simple hing-loss to do this task. Squre 
    Dtype* theta = bottom[0]->mutable_cpu_data(); // predicted
    Dtype* loss_array = loss_.mutable_cpu_data();
    caffe_set(N_, Dtype(0), loss_array);
    int channels = bottom[0]->channels();
    const Dtype* gt_bias = bottom[1]->cpu_data();
    const int* gt_pos = pos_.cpu_data();
    for(int i=0; i<N_; i++)
    {
        Dtype mdist = Dtype(0);
        mdist = (theta[i*channels + gt_pos[0]] - gt_bias[2*i+0])*(theta[i*channels + gt_pos[0]] - gt_bias[2*i+0]);
        mdist += rate_hw_*rate_hw_*(theta[i*channels + gt_pos[1]] - gt_bias[2*i+1])*(theta[i*channels + gt_pos[1]] - gt_bias[2*i+1]);
        mdist = mdist - threshold_;
        if (mdist > 0)
        {
            loss_array[i] = mdist/2;
        }
        else
        {
            loss_array[i] = 0.0;
        }
    }
    Dtype loss = caffe_cpu_asum(N_, loss_array);
    loss /= N_;
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void StnKeyPointLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	/* not implemented */
	// CHECK(false) << "Error: not implemented.";
    const Dtype* theta = bottom[0]->cpu_data();
    Dtype* diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), diff);
    const Dtype* loss_array = loss_.cpu_data();
    int channels = bottom[0]->channels();
    const Dtype* gt_bias = bottom[1]->cpu_data();
    const int* gt_pos = pos_.cpu_data();
    for(int i=0; i<N_; i++)
    {
        if(loss_array[i] > 0)
        {
            int index = i*channels + gt_pos[0];
            diff[index] = theta[index] - gt_bias[2*i+0];
            index = i*channels + gt_pos[1];
            diff[index] = rate_hw_*(theta[index] - gt_bias[2*i+1]);
        }
    }
    caffe_cpu_scale(bottom[0]->count(), top[0]->cpu_diff()[0]/N_, diff, diff);
}

#ifdef CPU_ONLY
STUB_GPU(StnKeyPointLossLayer);
#endif

INSTANTIATE_CLASS(StnKeyPointLossLayer);
REGISTER_LAYER_CLASS(StnKeyPointLoss);

}  // namespace caffe
