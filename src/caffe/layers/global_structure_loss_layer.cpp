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
void GlobalStructureLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());
}

template <typename Dtype>
void GlobalStructureLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    this->D = bottom[0]->count()/bottom[0]->num();
    Dtype label = 0;
    Dtype index = 0;
    this->class_label.clear();
    int num = bottom[1]->num();
    for (int i=0; i<num; i++)
    {   
        label = bottom[1]->cpu_data()[i];
        if(this->class_label.count(label))
        {
            this->class_label[label][1] += 1;
        }
        else
        {
            vector<Dtype> tmp; 
            tmp.push_back(index);
            index += 1; // record the releative label
            tmp.push_back(1);
            this->class_label[label] = tmp;
        }
    }
    // reshape the varibals
    C = int(index);
    class_centers_.Reshape(C, D, 1, 1);
    class_centers_norm_.Reshape(C, 1, 1, 1);
    class_centers_product_.Reshape(C, D, D, 1);
    one_matrix_.Reshape(D, D, 1, 1);
    center_matrix_.Reshape(num, D, 1, 1);
    diff_xi_center_.Reshape(num, D, 1, 1);
    delta_flag_.Reshape(C, C, 1, 1);
    diff_centers_centers_.Reshape(C*C, D, 1, 1);
    sparse_codes_.Reshape(C, num, 1, 1);
    extend_vector_.Reshape(C, 1, 1, 1);
    caffe_set(C, Dtype(1.), extend_vector_.mutable_cpu_data());
    caffe_set(D*D, Dtype(0.), one_matrix_.mutable_cpu_data());
    for(int i=0; i<D; i++)
    {
        one_matrix_.mutable_cpu_data()[i*D + i] = 1;
    }
    top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void GlobalStructureLossLayer<Dtype>::Forward_cpu( 
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // the forward function
    // create the sparse codes
    int num = bottom[1]->num();
    int count = bottom[0]->count();
    Dtype label = 0;
    Dtype* data = sparse_codes_.mutable_cpu_data();
    Dtype* data1 = NULL;
    Dtype* data2 = NULL;
    Dtype* data3 = NULL;
    for(int i=0; i<num; i++)
    {   
        label = bottom[1]->cpu_data()[i];
        data[i + int(class_label[label][0])*num] = 1;
    }
    // compute the class mean value at batch_step
    caffe_set(class_centers_.count(), Dtype(0), class_centers_.mutable_cpu_data());
    for(int i=0; i<num; i++)
    {
        label = bottom[1]->cpu_data()[i];
        data1 = bottom[0]->mutable_cpu_data() + i*D;
        data2 = class_centers_.mutable_cpu_data() + int(class_label[label][0])*D;
        caffe_add(D, data1, data2, data2);
    }
    typename map<Dtype, vector<Dtype> >::iterator it;
    data2 = class_centers_norm_.mutable_cpu_data(); 
    for(it = class_label.begin(); it!=class_label.end(); it++)
    {   
        data1 = class_centers_.mutable_cpu_data() + int(it->second[0])*D;
        caffe_cpu_scale(D, Dtype(1.0/(it->second[1])), data1, data1);
        data2[int(it->second[0])] = caffe_cpu_dot(D, data1, data1);
    }
    caffe_powx(C, data2, Dtype(0.5), data2); // compute the L2 norm. 
    for(it = class_label.begin(); it!=class_label.end(); it++) // normalization the centers
    {
        data1 = class_centers_.mutable_cpu_data() + int(it->second[0])*D;
        caffe_cpu_scale(D, Dtype(1./data2[int(it->second[0])]), data1, data1);
    }
    // compute the (I-uc*uc')/|uc|
    for(it = class_label.begin(); it!=class_label.end(); it++)
    {
        data1 = class_centers_.mutable_cpu_data() + int(it->second[0])*D;
        data2 = class_centers_product_.mutable_cpu_data() + int(it->second[0])*D*D;
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, D, D, 1, 
            Dtype(1.0), data1, data1, Dtype(0), data2);
        caffe_sub(D*D, one_matrix_.mutable_cpu_data(), data2, data2);
        Dtype norm = class_centers_norm_.mutable_cpu_data()[int(it->second[0])];
        caffe_cpu_scale(D*D, Dtype(1./norm), data2, data2);
    }
    // create the N*D center matrix
    data1 = sparse_codes_.mutable_cpu_data();
    data2 = class_centers_.mutable_cpu_data();
    data3 = center_matrix_.mutable_cpu_data();
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, num, D, C, 
        Dtype(1), data1, data2, Dtype(0), data3); // need to be done
    // create the diff_xi_center_ matrix
    data1 = bottom[0]->mutable_cpu_data();
    data2 = center_matrix_.mutable_cpu_data();
    data3 = diff_xi_center_.mutable_cpu_data();
    caffe_sub(count, data1, data2, data3);
    // create the diff_centers_centers_
    data1 = class_centers_.mutable_cpu_data();
    data2 = extend_vector_.mutable_cpu_data();
    data3 = diff_centers_centers_.mutable_cpu_data();
    // copy the centers with C times, use each center to sub this matrix
    for (int i=0; i<C; i++)
    {
        // copy
        caffe_copy(C*D, data1, &data3[i*C*D]);
        // utilize the C to fast compute each center's difference with other centers
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, C, D, 1,
            Dtype(1), data2, &data1[i*D], Dtype(-1), &data3[i*C*D]);
    }
    // compute the loss
    float margin = this->layer_param_.global_structure_loss_param().margin();
    // record the loss in forward for each pair of centers
    data1 = delta_flag_.mutable_cpu_data(); 
    data2 = diff_centers_centers_.mutable_cpu_data();
    // could be optimized with half computation and then copy 
    Dtype loss_inter = 0;
    for (int i=0; i<C-1; i++)
    {
        for(int j=i+1; j<C; j++)
        {
            int offset = (i*C+j)*D;
            Dtype tmp = caffe_cpu_dot(D, data2+offset, data2+offset);
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
    data1 = diff_xi_center_.mutable_cpu_data(); // the data that has been mean subtracted
    data2 = bottom[1]->mutable_cpu_data(); // the label 
    for(int i=0; i<num; i++)
    {
        int offset = i*D;
        Dtype tmp = caffe_cpu_dot(D, data1+offset, data1+offset);
        loss_intra += tmp/class_label[data2[i]][1];
    }
    loss_intra = loss_intra/2/C;
    Dtype weight = this->layer_param_.global_structure_loss_param().weight();
    Dtype loss = loss_intra + weight*loss_inter;
    top[0]->mutable_cpu_data()[0] = loss;
    // top[0]->mutable_cpu_data()[1] = loss_intra;
    // top[0]->mutable_cpu_data()[2] = loss_inter;
}

template <typename Dtype>
void GlobalStructureLossLayer<Dtype>::Backward_cpu( const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // the backward function
    // using cached diff_xi_center_, diff_centers_centers_ and  delta_flag_ to update the diff
    Dtype weight = this->layer_param_.global_structure_loss_param().weight();
    Dtype loss_weight = top[0]->cpu_diff()[0];
    // update the diff from the intra loss
    int num = bottom[0]->num();
    Dtype* data1 = bottom[1]->mutable_cpu_data(); // label, ok
    Dtype* data2 = diff_xi_center_.mutable_cpu_data(); // the intra diff, ok
    Dtype* data3 = bottom[0]->mutable_cpu_diff(); // the diff data
    // reset the delta_flag_ with 0 and 1
    for(int i=0; i<C; i++)
        for(int j=0; j<C; j++)
            delta_flag_.mutable_cpu_data()[i*C+j] = delta_flag_.mutable_cpu_data()[i*C+j] > 0 ? 1 : 0;
    Dtype* data4 = delta_flag_.mutable_cpu_data(); // ok
    Dtype* data5 = diff_centers_centers_.mutable_cpu_data(); // ok
    Dtype* data6 = class_centers_product_.mutable_cpu_data(); //
    Blob<Dtype> tmp;
    tmp.Reshape(1, D, 1, 1);
    // set the update
    for(int i=0; i<num; i++)
    {
        int label = int(class_label[data1[i]][0]);
        Dtype nc = class_label[data1[i]][1];
        int offset = i*D;
        caffe_set(D, Dtype(0), tmp.mutable_cpu_data());
        // update the diff from the intra loss
        for(int j=0; j<num; j++) // could be speed due to the useless computation for the same class
        {
            if(data1[j] == data1[i])
                caffe_add(D, data2+j*D, tmp.mutable_cpu_data(), tmp.mutable_cpu_data());
        }
        caffe_copy(D, data2+offset, data3+offset);
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, D, D, 
            Dtype(-1/nc), tmp.mutable_cpu_data(), data6+label*D*D, Dtype(1.0), data3+offset);
        // update the diff from the inter loss
        Dtype scale = 0;
        if (C > 1)
            scale = weight*(-2)/(C-1);
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, D, C,
            scale, data4+label*C, data5+label*C*D, Dtype(0.0), tmp.mutable_cpu_data());
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, D, D,
            Dtype(1.0), tmp.mutable_cpu_data(), data6+label*D*D, Dtype(1.0), data3+offset);
        // scale the loss, additional with the loss weight
        caffe_cpu_scale(D, Dtype(loss_weight/C/class_label[data1[i]][1]), data3+offset, data3+offset);
    }
}   

#ifdef CPU_ONLY
    STUB_GPU(GlobalStructureLossLayer);
#endif
INSTANTIATE_CLASS(GlobalStructureLossLayer);
REGISTER_LAYER_CLASS(GlobalStructureLoss);
} // end namespace of caffe
