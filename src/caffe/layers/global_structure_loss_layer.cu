/*
*
*=============================================
* for batch based global structure loss
*
* dangweili@gmail.com
*
*============================================
*/
#include <vector>

#include "caffe/layers/global_structure_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GlobalStructureLossLayer<Dtype>::Forward_gpu( 
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // the forward function
    // create the sparse codes
    int num = bottom[1]->num();
    int count = bottom[0]->count();
    Dtype label(0);
    Dtype* data1 = NULL;
    Dtype* data2 = NULL;
    Dtype* data3 = NULL;
    data1 = sparse_codes_.mutable_cpu_data();
    data2 = bottom[1]->mutable_cpu_data();
    for(int i=0; i<num; i++)
    {   
        label = data2[i];
        data1[i + int(class_label[label][0])*num] = 1;
    }
    // compute the class mean value
    caffe_set(class_centers_.count(), Dtype(0), class_centers_.mutable_cpu_data());
    data1 = bottom[1]->mutable_cpu_data();
    for(int i=0; i<num; i++)
    {
        label = data1[i];
        data2 = bottom[0]->mutable_gpu_data() + i*D;
        data3 = class_centers_.mutable_gpu_data() + int(class_label[label][0])*D;
        caffe_gpu_add(D, data2, data3, data3);
    }
    typename map<Dtype, vector<Dtype> >::iterator it;
    for(it = class_label.begin(); it!=class_label.end(); it++)
    {   
        data1 = class_centers_.mutable_gpu_data() + int(it->second[0])*D;
        caffe_gpu_scale(D, Dtype(1.0/(it->second[1])), data1, data1);
    }
    // create the N*D center matrix
    data1 = sparse_codes_.mutable_gpu_data();
    data2 = class_centers_.mutable_gpu_data();
    data3 = center_matrix_.mutable_gpu_data();
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, num, D, C, 
        Dtype(1), data1, data2, Dtype(0), data3); // need to be done
    // create the diff_xi_center_ matrix
    data1 = bottom[0]->mutable_gpu_data();
    data2 = center_matrix_.mutable_gpu_data();
    data3 = diff_xi_center_.mutable_gpu_data();
    caffe_gpu_sub(count, data1, data2, data3);
    // create the diff_centers_centers_
    data1 = class_centers_.mutable_gpu_data();
    data2 = extend_vector_.mutable_gpu_data();
    data3 = diff_centers_centers_.mutable_gpu_data();
    // copy the centers with C times, use each center to sub this matrix
    for (int i=0; i<C; i++)
    {
        // copy
        caffe_copy(C*D, data1, &data3[i*C*D]);
        // utilize the C to fast compute each center's difference with other centers
        caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, D, C, 1,
            Dtype(1), &data1[i*D], data2, Dtype(-1), &data3[i*C*D]);
    }
    // compute the loss
    float margin = this->layer_param_.global_structure_loss_param().margin();
    // record the loss in forward for each pair of centers
    data1 = delta_flag_.mutable_cpu_data(); 
    data2 = diff_centers_centers_.mutable_gpu_data();
    // could be optimized with half computation and then copy 
    Dtype loss_inter = 0;
    for (int i=0; i<C-1; i++)
    {
        for(int j=i+1; j<C; j++)
        {
            int offset = (i*C+j)*D;
            Dtype tmp = 0;
            caffe_gpu_dot(D, data2+offset, data2+offset, &tmp);
            Dtype mdist = std::max(Dtype(0), margin - tmp);
            loss_inter += mdist;
            data1[i*C + j] = mdist;
            data1[j*C + i] = mdist;
        }
    }
    if (C > 1)
        loss_inter = loss_inter*2/C/(C-1);
    else
        loss_inter = 0;
    // compute the intra loss
    Dtype loss_intra = 0;
    data1 = diff_xi_center_.mutable_gpu_data(); // the data that has been mean subtracted
    data2 = bottom[1]->mutable_cpu_data(); // the label 
    for(int i=0; i<num; i++)
    {
        int offset = i*D;
        Dtype tmp = 0;
        caffe_gpu_dot(D, data1+offset, data1+offset, &tmp);
        loss_intra += tmp/class_label[data2[i]][1];
    }
    loss_intra = loss_intra/2/C;
    Dtype weight = this->layer_param_.global_structure_loss_param().weight();
    Dtype loss = loss_intra + weight*loss_inter;
    top[0]->mutable_cpu_data()[0] = loss;
    top[0]->mutable_cpu_data()[1] = loss_intra;
    top[0]->mutable_cpu_data()[2] = loss_inter;
}

template <typename Dtype>
void GlobalStructureLossLayer<Dtype>::Backward_gpu( const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // the backward function
    // using cached diff_xi_center_, diff_centers_centers_ and  delta_flag_ to update the diff
    Dtype weight = this->layer_param_.global_structure_loss_param().weight();
    Dtype loss_weight = top[0]->cpu_diff()[0];
    // update the diff from the intra loss
    int num = bottom[0]->num();
    Dtype* data1 = bottom[1]->mutable_cpu_data(); // label
    Dtype* data2 = diff_xi_center_.mutable_gpu_data(); // the intra diff
    Dtype* data3 = bottom[0]->mutable_gpu_diff(); // the diff data
    // reset the delta_flag_ with 0 and 1
    for(int i=0; i<C; i++)
        for(int j=0; j<C; j++)
            delta_flag_.mutable_cpu_data()[i*C+j] = delta_flag_.mutable_cpu_data()[i*C+j] > 0 ? 1 : 0;
    Dtype* data4 = delta_flag_.mutable_gpu_data();
    Dtype* data5 = diff_centers_centers_.mutable_gpu_data();
    // set the update
    for(int i=0; i<num; i++)
    {
        int label = int(class_label[data1[i]][0]);
        int offset = i*D;
        // update the diff from the intra loss
        caffe_copy(D, data2+offset, data3+offset);
        // update the diff from the inter loss
        Dtype scale = 0;
        if (C > 1)
            scale = weight*(-2)/(C-1);
        caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, D, C,
            scale, data4+label*C, data5+label*C*D, Dtype(1.0), data3+offset);
        // scale the loss
        caffe_gpu_scale(D, Dtype(loss_weight/C/class_label[data1[i]][1]), data3+offset, data3+offset);
    }
}   

INSTANTIATE_LAYER_GPU_FUNCS(GlobalStructureLossLayer);
} // end namespace of caffe
