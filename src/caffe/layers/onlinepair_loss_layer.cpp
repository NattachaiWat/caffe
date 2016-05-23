#include <algorithm>
#include <vector>


#include "caffe/layers/onlinepair_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/other_functions.hpp"

namespace caffe {

template <typename Dtype>
void OnlinePairLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_GE(bottom[1]->num(), 2);
  // malloc the data for diff_
  diff_.Reshape(bottom[0]->channels(), 1, 1, 1);
  // store the information about each channel
  // general the pair of pos is less than 
  pairdist_neg_.reserve(512*512);
  pairdist_pos_.reserve(512*512);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
  top[0]->Reshape(1,1,1,1);

}
template <typename Dtype>
void OnlinePairLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // int count = bottom[0]->count();
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.onlinepair_loss_param().margin();
  bool legacy_version =
      this->layer_param_.onlinepair_loss_param().legacy_version();
  int hards_pos = this->layer_param_.onlinepair_loss_param().hards_pos();
  int hards_neg = this->layer_param_.onlinepair_loss_param().hards_neg();
  // forward, step1, compute the distance matrix
  int num = bottom[0]->num();
  PairDist tmp;
  pairdist_neg_.clear();
  pairdist_pos_.clear();
  for(int i=0; i<num-1; i++)
    for (int j=i+1; j<num; j++)
    {
       caffe_sub(
            channels, 
            bottom[0]->cpu_data() + i*channels, 
            bottom[0]->cpu_data() + j*channels, 
            diff_.mutable_cpu_data());
       tmp.dist = caffe_cpu_dot(channels, diff_.cpu_data(), diff_.cpu_data());
       tmp.first = i;
       tmp.second = j;
       tmp.flag = 
            bottom[1]->mutable_cpu_data()[i] == bottom[1]->mutable_cpu_data()[j] ? 1 : 0;
       if( tmp.flag == 1)
       {    pairdist_pos_.push_back( tmp ); }
       else
       {    pairdist_neg_.push_back( tmp ); }
    }
  // sort function
  
  // std::sort(pairdist_pos_.begin(), pairdist_pos_.end(), pair_descend);
  // std::sort(pairdist_neg_.begin(), pairdist_neg_.end(), pair_ascend);
  if ( pairdist_pos_.size() > 0)
  {
    std::qsort(&pairdist_pos_[0], pairdist_pos_.size(), sizeof(PairDist), pair_descend_qsort);
  }
  if ( pairdist_neg_.size() > 0)
  {
    std::qsort(&pairdist_neg_[0], pairdist_neg_.size(), sizeof(PairDist), pair_ascend_qsort);
  }


  // sort_pairdist(pairdist_pos_, false);
  // sort_pairdist(pairdist_neg_, true);
  // take the first hards elements for backward and loss compution
  int pos_num = pairdist_pos_.size() > hards_pos ? hards_pos : pairdist_pos_.size();
  int neg_num = pairdist_neg_.size() > hards_neg ? hards_neg : pairdist_neg_.size();

  //cout the num information
  //std::cout << "pos_num: " << pos_num << std::endl;
  //std::cout << "neg_num: " << neg_num << std::endl;
  //std::cout << "hards_pos_num: " << hards_pos << std::endl;
  //std::cout << "hards_neg_num: " << hards_neg << std::endl;
  //std::cout << "pairdist_pos_.size: " << pairdist_pos_.size()  << std::endl;
  //std::cout << "pairdist_neg_.size " << pairdist_neg_.size() << std::endl;

  Dtype loss(0.0);
  for (int i = 0; i<pos_num; i++)
  {
    loss += pairdist_pos_[i].dist; 
  }
  for (int i = 0; i < neg_num; i++)
  {
    if (legacy_version)
    {
        loss += std::max( Dtype(0.0), margin - pairdist_neg_[i].dist);
    }
    else
    {
        Dtype dist = std::max( Dtype(0.0), margin - sqrt( pairdist_neg_[i].dist) );
        loss += dist * dist;
    }
  }
  loss = loss / Dtype(pos_num + neg_num) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
void OnlinePairLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
  int first = 0;
  int second = 0;
  caffe_set(count, Dtype(0), bottom[0]->mutable_cpu_diff());
  // get the gradient point, which is used to set gradient value
  Dtype* bout = bottom[0]->mutable_cpu_diff();

  for (int i=0; i< pos_num; i++)
  {
    first = pairdist_pos_[i].first;
    second = pairdist_pos_[i].second;
    // compute the a-b, first - second
    caffe_sub(
        channels, 
        bottom[0]->cpu_data() + first*channels, 
        bottom[0]->cpu_data() + second*channels, 
        diff_.mutable_cpu_data());
    for(int j = 0; j < 2; j++)
    {
        Dtype sign = (j == 0) ? 1 : -1; 
        int position = (j == 0)? first : second;
        Dtype alpha = sign * top[0]->cpu_diff()[0] / Dtype(pos_num + neg_num);
        // update the two example's gradient
        caffe_cpu_axpby(
            channels,
            alpha,
            diff_.cpu_data(),
            Dtype(1.0),
            bout + (position*channels));
     }
  }
  for (int i=0; i< neg_num; i++)
  {
    first = pairdist_neg_[i].first;
    second = pairdist_neg_[i].second;
    // compute the a-b, first - second
    caffe_sub(
        channels, 
        bottom[0]->cpu_data() + first*channels, 
        bottom[0]->cpu_data() + second*channels, 
        diff_.mutable_cpu_data());
    for (int j=0; j<2; j++)
    {
        Dtype sign = (j == 0) ? 1 : -1; 
        int position = (j == 0) ? first : second;
        Dtype alpha = sign * top[0]->cpu_diff()[0] / Dtype(pos_num + neg_num);
        Dtype mdist(0.0);
        Dtype beta(0.0);
        if (legacy_version) {
            mdist = margin - pairdist_neg_[i].dist;
            beta = -alpha;
        }
        else{
            Dtype dist = sqrt(pairdist_neg_[i].dist);
            mdist = margin - dist;
            beta  = -alpha * mdist / (dist + Dtype(1e-4));
        }
        if ( mdist > Dtype(0.0) )
        {
            caffe_cpu_axpby(
                channels,
                beta,
                diff_.cpu_data(),
                Dtype(1.0),
                bout + (position*channels));
        }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(OnlinePairLossLayer);
#endif

INSTANTIATE_CLASS(OnlinePairLossLayer);
REGISTER_LAYER_CLASS(OnlinePairLoss);

}  // namespace caffe
