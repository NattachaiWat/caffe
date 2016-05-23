#include <algorithm>
#include <vector>

#include "caffe/layers/onlinepair_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>
#include "caffe/util/other_functions.hpp"

namespace caffe {

template <typename Dtype>
void OnlinePairLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // const int count = bottom[0]->count();
  int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.onlinepair_loss_param().margin();
  bool legacy_version =
      this->layer_param_.onlinepair_loss_param().legacy_version();
  int hards_pos = this->layer_param_.onlinepair_loss_param().hards_pos();
  int hards_neg = this->layer_param_.onlinepair_loss_param().hards_neg();
  // forward, step1, compute the distance matrix
  PairDist tmp;
  pairdist_neg_.clear();
  pairdist_pos_.clear();
  for (int i = 0; i<num-1; i++)
    for (int j = i+1; j < num; j++)
    {
        caffe_gpu_sub(
            channels, 
            bottom[0]->gpu_data() + (i*channels),
            bottom[0]->gpu_data() + (j*channels),
            diff_.mutable_gpu_data());
        Dtype dist_tmp;
        caffe_gpu_dot(channels, diff_.cpu_data(), diff_.cpu_data(), &(dist_tmp));
        tmp.dist = dist_tmp; 
        tmp.first = i;
        tmp.second = j;
        tmp.flag = 
            bottom[1]->mutable_cpu_data()[i] == bottom[1]->mutable_cpu_data()[j] ? 1 : 0;
        if (tmp.flag == 1)
        { pairdist_pos_.push_back( tmp ); }
        else
        { pairdist_neg_.push_back( tmp ); }
    }

  // sort the distance matrix
  std::sort( pairdist_pos_.begin(), pairdist_pos_.end(), pair_descend);
  std::sort( pairdist_neg_.begin(), pairdist_neg_.end(), pair_ascend);
  // sort_pairdist(pairdist_pos_, false);
  // sort_pairdist(pairdist_neg_, true);
  // take the first hards elements for backward and loss computation
  int pos_num = pairdist_pos_.size() > hards_pos ? hards_pos : pairdist_pos_.size();
  int neg_num = pairdist_neg_.size() > hards_neg ? hards_neg : pairdist_neg_.size();
  Dtype loss(0.0);
  for (int i = 0; i<pos_num; i++)
  {
    loss += pairdist_pos_[i].dist;
  }
  for (int i = 0; i<neg_num; i++)
  {
    if (legacy_version) 
    {
        loss += std::max( Dtype(0.0), margin - pairdist_neg_[i].dist);
    }
    else
    {
        Dtype dist = std::max( Dtype(0.0), margin - sqrt( pairdist_neg_[i].dist ) );
        loss += dist * dist;
    }
  }
  loss = loss / Dtype(pos_num + neg_num) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void OnlinePairLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.onlinepair_loss_param().margin();
  bool legacy_version =
    this->layer_param_.onlinepair_loss_param().legacy_version();
  int count = bottom[0]->count();
  const int channels = bottom[0]->channels();
  int hards_pos = this->layer_param_.onlinepair_loss_param().hards_pos();
  int hards_neg = this->layer_param_.onlinepair_loss_param().hards_neg();
  int pos_num = pairdist_pos_.size() > hards_pos ? hards_pos : pairdist_pos_.size();
  int neg_num = pairdist_neg_.size() > hards_neg ? hards_neg : pairdist_neg_.size();
  // the backward is here, first for similar and then for dissimilar pairs
  // before take all the gradient, reset it with zeros
  caffe_gpu_set(count, Dtype(0), bottom[0]->mutable_gpu_diff());
  int first;
  int second;
  // get the pointer of backward gradient
  Dtype* bout = bottom[0]->mutable_cpu_diff();
  std::cout<< "start the pos gradient computation" << std::endl;
  std::cout<< "pos count is : " << pos_num << std::endl;
  for (int i = 0; i < pos_num; i++)
  {
      first = pairdist_pos_[i].first;
      second = pairdist_pos_[i].second;
      // compute the diff 
      caffe_gpu_sub(
        channels,
        bottom[0]->gpu_data() + first*channels,
        bottom[0]->gpu_data() + second*channels,
        diff_.mutable_gpu_data());
      for (int j = 0; j < 2; j++)
      {
        Dtype sign = (j == 0) ? 1 : -1;
        int position = (j == 0)? first : second;
        Dtype alpha = sign * top[0]->cpu_diff()[0] / Dtype(pos_num + neg_num);
        // update the gradient
        caffe_gpu_axpby(
            channels,
            alpha,
            diff_.gpu_data(),
            Dtype(1.0),
            bout + (position*channels));
      }
  }
  std::cout<< "start the neg gradient computation" << std::endl;
  std::cout<< "neg count is : " << neg_num << std::endl;
  for (int i = 0; i < neg_num; i ++)
  {
    first = pairdist_neg_[i].first;
    second = pairdist_neg_[i].second;
    // compute the a-b, first - second
    caffe_gpu_sub(
        channels,
        bottom[0]->gpu_data() + first*channels,
        bottom[0]->gpu_data() + second*channels,
        diff_.mutable_gpu_data());
    for (int j = 0; j < 2; j++)
    {
        Dtype sign = (j == 0) ? 1 : -1;
        int position = (j == 0)? first : second;
        Dtype alpha = sign * top[0]->cpu_diff()[0] / Dtype(pos_num + neg_num);
        Dtype mdist(0.0);
        Dtype beta(0.0);
        if (legacy_version) {
            mdist = margin - pairdist_neg_[i].dist;
            beta = -alpha;
        }
        else {
            Dtype dist = sqrt(pairdist_neg_[i].dist);
            mdist = margin - dist;
            beta  = -alpha * mdist / (dist + Dtype(1e-4));
        }
        if ( mdist > Dtype(0.0) )
        {
            caffe_gpu_axpby(
                channels,
                beta,
                diff_.gpu_data(),
                Dtype(1.0),
                bout + (position*channels));
        }
    }
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(OnlinePairLossLayer);

}  // namespace caffe
