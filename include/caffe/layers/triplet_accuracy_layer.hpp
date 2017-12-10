#ifndef CAFFE_TRIPLET_ACCURACY_LAYER_HPP_
#define CAFFE_TRIPLET_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

// triplet accuracy layer
template <typename Dtype>
class TripletAccuracyLayer: public Layer<Dtype> {
 public:
  explicit TripletAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TripletAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Dtype margin_; // the margin of loss
  Blob<Dtype> diff_ap_; // a - p
  Blob<Dtype> diff_an_; // a - n
  Blob<Dtype> summer_vec_num_; // for cpu accuracy
  Blob<Dtype> summer_vec_channel_; // for gpu accuracy
  Blob<Dtype> dist_sq_; // totally n elements, ||a-p||2  - ||a-n||2
  Blob<Dtype> dist_sq_ap_; // the sum of each channel for examples a-p
  Blob<Dtype> dist_sq_an_; // the sum of each channel for examples of a-n
};

}  // namespace caffe

#endif  // CAFFE_TRIPLET_ACCURACY_LAYER_HPP_
