#ifndef CAFFE_GLOBAL_STRUCTURE_LOSS_LAYER_HPP_
#define CAFFE_GLOBAL_STRUCTURE_LOSS_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"


namespace caffe {
template <typename Dtype>
class GlobalStructureLossLayer: public LossLayer<Dtype> {
    public:
        explicit GlobalStructureLossLayer(const LayerParameter& param)
            : LossLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>&top);
        // virtual void Reshape(
        //    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        // virtual inline int ExactNumBottomBlobs() const { return -1; }
        virtual inline int MinBottomBlobs() const {return 2; }
        virtual inline int MaxBottomBlobs() const {return 2; }
        virtual inline const char* type() const {return "GlobalStructureLoss"; }
        virtual inline bool AutoTopBlobs() const {return true; } 
        
        virtual inline bool AllowForceBackward(const int bottom_index) const {
            return bottom_index != 1;
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

        // the protected variables

        Blob<Dtype> class_centers_; // dictary, C*D*1*1 cache for label centers
        Blob<Dtype> center_matrix_; // tmp, N*D to speed mean subtracted
        Blob<Dtype> sparse_codes_; // C*N, to generate the code for other operator
        // N*D*1*1, the differ of x and it's center,  cache for backward propagation
        Blob<Dtype> diff_xi_center_; 
        Blob<Dtype> delta_flag_; // C*C with diagonal value equels zero, cache for backward
        Blob<Dtype> diff_centers_centers_; // C^2*D, with each value is the diff between two centers 
        map<Dtype, vector<Dtype> > class_label; // unique class label for each mini-batch
        Blob<Dtype> extend_vector_; // for generate the C^2*D matrix
        int C; // totally C classes
        int D; // the feature dimension
}; 

}   // end namespace of caffe


#endif // CAFFE_GLOBAL_STRUCTURE_LOSS_LAYER_HPP_

