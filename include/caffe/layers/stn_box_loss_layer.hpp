#ifndef STN_BOX_LOSS_LAYERS_HPP_
#define STN_BOX_LOSS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/* Input: theta and threshold, which contraints the bounding box in the image.
    Now, it only support the fixed 4 parameters, with scale and transformer.
 * Output: loss, one single value
*/

template <typename Dtype>
class StnBoxLossLayer : public LossLayer<Dtype> {
public:
  explicit StnBoxLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "StnBoxLoss"; }

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
  int N; // the batch_size
  Dtype threshold; // threshold for each point
  // vector<int> pos; // useness
  Blob<Dtype> loss_; // record each example's loss
};

}  // namespace caffe

#endif  // STN_BOX_LOSS_LAYERS_HPP_
