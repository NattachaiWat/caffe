#include <algorithm>
#include <vector>

#include "caffe/layers/contrastive_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ContrastiveAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
  top[0]->Reshape(1,3,1,1);
}
template <typename Dtype>
void ContrastiveAccuracyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}


template <typename Dtype>
void ContrastiveAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int pos_cnt = 0;
  int neg_cnt = 0;
  int pos_right = 0;
  int neg_right = 0;
  float eps = 0.0001;

  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.contrastive_accuracy_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_accuracy_param().legacy_version();
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      // handle the pos pairs
      pos_cnt += 1;
      if (dist_sq_.cpu_data()[i] < margin) pos_right += 1;

      // loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      if (legacy_version) {
        // handle the neg pairs
        neg_cnt += 1;
        if( std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0)) == 0)
        {
            neg_right += 1;
        }

        // loss += std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0));
      } else {
        Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq_.cpu_data()[i]),
          Dtype(0.0));
        // loss += dist*dist;
        // handle the neg pairs
        neg_cnt += 1;
        if (dist == 0)
        {   
            neg_right += 1;
        }
      }
    }
  }
  float pos_accuracy = pos_right/(pos_cnt + eps);
  float neg_accuracy = neg_right/(neg_cnt + eps);
  float accuracy = 0.5*(pos_accuracy + neg_accuracy);
  top[0]->mutable_cpu_data()[0] = accuracy;
  top[0]->mutable_cpu_data()[1] = pos_accuracy;
  top[0]->mutable_cpu_data()[2] = neg_accuracy;
}


#ifdef CPU_ONLY
STUB_GPU(ContrastiveAccuracyLayer);
#endif

INSTANTIATE_CLASS(ContrastiveAccuracyLayer);
REGISTER_LAYER_CLASS(ContrastiveAccuracy);

}  // namespace caffe
