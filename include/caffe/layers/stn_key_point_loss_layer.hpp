#ifndef STN_CENTER_LOSS_LAYERS_HPP_
#define STN_CENTER_LOSS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/* Input: theta and center
 * Output: loss, one single value
*/

template <typename Dtype>
class StnKeyPointLossLayer : public LossLayer<Dtype> {
public:
  explicit StnKeyPointLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "StnKeyPointLoss"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index<1;
  }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  int N_;
  Dtype threshold_;
  Dtype rate_hw_;
  Blob<int> pos_;

  Blob<Dtype> loss_; // for speed
};

}  // namespace caffe

#endif  // STN_CENTER_LOSS_LAYERS_HPP_
