#ifndef STN_POS_LOSS_LAYERS_HPP_
#define STN_POS_LOSS_LAYERS_HPP_

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
class StnPosLossLayer : public LossLayer<Dtype> {
public:
  explicit StnPosLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "StnPosLoss"; }

  virtual inline int ExactNumBottomBlobs() const { return 1; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
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
  int N;
  Dtype threshold; // theta >= threshold
  Blob<int> pos_; // this is the mask, only the pos should satify the constraints
  Blob<Dtype> loss_; // only record and back the pos's gradient
};

}  // namespace caffe

#endif  // STN_POS_LOSS_LAYERS_HPP_
