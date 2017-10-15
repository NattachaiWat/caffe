#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/multiimage_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "opencv2/imgproc/imgproc.hpp"
namespace caffe {

template <typename Dtype>
MultiImageDataLayer<Dtype>::~MultiImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void MultiImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.multiimage_data_param().new_height();
  const int new_width = this->layer_param_.multiimage_data_param().new_width();
  const bool is_color  = this->layer_param_.multiimage_data_param().is_color();
  string root_folder = this->layer_param_.multiimage_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.multiimage_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
/*  int label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }
*/
  string line;
  int line_num = 1;
  int num_labels = 0;
  while( std::getline(infile, line) )
  {
      std::istringstream iss(line);
      vector<int> label;
      int label_temp;
      iss >> filename; // read the image name
      while( iss>> label_temp ) label.push_back( label_temp );
      if( line_num == 1) num_labels = label.size();
      CHECK_EQ( label.size(), num_labels) << filename << std::endl << 
        "All image should has equel label" << std::endl;
      line_num++;
      lines_.push_back( std::make_pair( filename, label) );
  }
  // generally there is no shuffle
  if (this->layer_param_.multiimage_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.multiimage_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.multiimage_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.multiimage_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  // vector<int> label_shape(1, batch_size);
  if (this->output_labels_)
  {
    vector<int> label_shape(2);
    top[0]->Reshape(top_shape);
    label_shape[0] = batch_size;
    label_shape[1] = lines_[lines_id_].second.size();
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(label_shape);
    }     
  }
}

template <typename Dtype>
void MultiImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void MultiImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  MultiImageDataParameter multiimage_data_param = this->layer_param_.multiimage_data_param();
  const int batch_size = multiimage_data_param.batch_size();
  const int new_height = multiimage_data_param.new_height();
  const int new_width = multiimage_data_param.new_width();
  const bool is_color = multiimage_data_param.is_color();
  string root_folder = multiimage_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
    
  // load the pca jittering and rotation parameter
  float rotation_theta_min = multiimage_data_param.rotation_theta_min();  
  float rotation_theta_max = multiimage_data_param.rotation_theta_max();
  bool flag_rotation = 0;
  if (rotation_theta_min ==0 && rotation_theta_max == 0)
  {
    flag_rotation = 0;
  }else
  {
    flag_rotation = 1;
  }
  int eig_value_size = multiimage_data_param.eig_value_size();
  int eig_vector_size = multiimage_data_param.eig_vector_size();
  vector<float> eig_value;
  vector<float> eig_vector;
  float pca_jitter_mean = multiimage_data_param.eig_coef_mean();
  float pca_jitter_var = multiimage_data_param.eig_coef_var();
  bool flag_pca_jitter = 0;
  float pca_jitter_coef[3];
  float temp_jitter[3];
  if( eig_value_size == 3  && eig_vector_size == 9)
  {
    flag_pca_jitter = 1;
    for(int i=0; i<eig_value_size; i++) eig_value.push_back( multiimage_data_param.eig_value(i) );
    for(int i=0; i<eig_vector_size; i++) eig_vector.push_back( multiimage_data_param.eig_vector(i) );
  }
 
  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    // load image, pca jittering, rotation; sub mean, scale and crop
    // 1 load image
    cv::Mat cv_img_temp_ = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    cv::Mat cv_img_temp;
    cv_img_temp_.convertTo(cv_img_temp, CV_32F);
    cv::Mat cv_img_pca;
    // 2 pca jittering
    if (flag_pca_jitter)
    {
        caffe_rng_gaussian(3, pca_jitter_mean, pca_jitter_var, pca_jitter_coef); 
        for(int i_ = 0; i_<3; i_++)
        {
            temp_jitter[i_] = 0;
            for(int j_ = 0; j_<3; j_++)
            {
                temp_jitter[i_] += pca_jitter_coef[j_] * eig_value[j_] * eig_vector[i_*3 + j_];
            }
        }
        cv_img_pca = cv_img_temp + cv::Scalar(temp_jitter[0], temp_jitter[1], temp_jitter[2]);
    }else
    {
        cv_img_pca = cv_img_temp;
    }
    // 3 rotation
    cv::Mat cv_img_rotation;
    if (flag_rotation)
    {
        cv::Point center = cv::Point(cv_img_pca.cols/2, cv_img_pca.rows/2);
        float angle = 0;
        caffe_rng_uniform(1, rotation_theta_min, rotation_theta_max, &angle);
        cv::Mat rot_mat = cv::getRotationMatrix2D( center, double(angle), double(1.0) );
        cv::warpAffine(cv_img_pca, cv_img_rotation, rot_mat, cv_img_pca.size());
    }else
    {
        cv_img_rotation = cv_img_pca;
    } 
    cv::Mat cv_img = cv_img_rotation;

    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    // prefetch_label[item_id] = lines_[lines_id_].second;
    if( this->output_labels_ )
    {
        int label_cnt = lines_[lines_id_].second.size();
        for(int label_index = 0; label_index < label_cnt; label_index++)
        {
            prefetch_label[item_id*label_cnt + label_index] = lines_[lines_id_].second[label_index];
        }
    }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.multiimage_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MultiImageDataLayer);
REGISTER_LAYER_CLASS(MultiImageData);

}  // namespace caffe
#endif  // USE_OPENCV
