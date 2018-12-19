#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/intra_class_center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void IntraClassCenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const int num_output = this->layer_param_.itra_class_center_loss_param().num_output();  
  N_ = num_output;      // 输出神经元的个数
  const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.itra_class_center_loss_param().axis());        // 特征输入长度,将特征C*H*W拉直为一个特征
  K_ = bottom[0]->count(axis);

  if (this->blobs_.size() > 0) {        // blob中存储的是每次更新的类中心值
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> center_shape(2);
    center_shape[0] = N_;
    center_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(center_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(this->layer_param_.itra_class_center_loss_param().center_filler()));
    center_filler->Fill(this->blobs_[0].get());

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void IntraClassCenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  M_ = bottom[0]->num();    // mini_batch
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  LossLayer<Dtype>::Reshape(bottom, top);
  variation_sum_.ReshapeLike(*this->blobs_[0]);
  intra_class_diff_.Reshape(M_, K_, 1, 1);
  intra_class_dist_sq_.Reshape(M_, 1, 1, 1);
}

template <typename Dtype>
void IntraClassCenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* center = this->blobs_[0]->cpu_data();
  Dtype* intra_class_diff = intra_class_diff_.mutable_cpu_data(); //记录类内距离
  Dtype margin = this->layer_param_.itra_class_center_loss_param().margin();  // margin
  Dtype loss(0.0);
  
  for (int i = 0; i < M_; i++){     // 第i个样本
      const int label_value = static_cast<int>(label[i]);  // 第i个样本对应的label标签
      caffe_sub(K_, bottom_data + i * K_, center + label_value * K_, intra_class_diff + (i * K_));
      intra_class_dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(K_, intra_class_diff_.cpu_data()+i * K_, intra_class_diff_.cpu_data()+i * K_);
      Dtype dist = std::max<Dtype>(sqrt(intra_class_dist_sq_.mutable_cpu_data()[i]) - margin,Dtype(0.0));
      loss += dist*dist;
  }
  loss = loss / M_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss; 
}

template <typename Dtype>
void IntraClassCenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // Gradient with respect to centers   // 更新类中心
  if (this->param_propagate_down_[0]) {
    const Dtype* label = bottom[1]->cpu_data();   // label
    Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();   // center更新值
    Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();
    const Dtype* intra_class_diff = intra_class_diff_.cpu_data();   // 里面记录了样本与类中心的距离

    caffe_set(N_ * K_, (Dtype)0., variation_sum_.mutable_cpu_data());
    for (int n = 0; n < N_; n++) {
      int count = 0;
      for (int m = 0; m < M_; m++) {
        const int label_value = static_cast<int>(label[m]);
        if (label_value == n) {
          count++;
          caffe_sub(K_, variation_sum_data + n * K_, intra_class_diff +  (m * K_), variation_sum_data + n * K_);
        }
      }
      caffe_axpy(K_, (Dtype)1./(count + (Dtype)1.), variation_sum_data + n * K_, center_diff + n * K_);
    }
  }
  
  // Gradient with respect to bottom data 
  Dtype* bottom_cpu_diff = bottom[0]->mutable_cpu_diff();
  const Dtype alpha = top[0]->cpu_diff()[0] / M_;
  Dtype margin = this->layer_param_.itra_class_center_loss_param().margin();  // margin
  const Dtype* intra_class_diff = intra_class_diff_.cpu_data();   // 里面记录了样本与类中心的距离
 
  for (int i = 0; i < M_; ++i){   // 第i个样本
      Dtype dist = sqrt(intra_class_dist_sq_.cpu_data()[i]);
      Dtype mdist = dist - margin;
      Dtype beta = -alpha * mdist / (dist + Dtype(1e-4));
      if(mdist > Dtype(0.0)){
        caffe_cpu_axpby(K_, beta, intra_class_diff + (i * K_),Dtype(1.0),bottom_cpu_diff + ( i * K_));
      }
  }
}

#ifdef CPU_ONLY
STUB_GPU(IntraClassCenterLossLayer);
#endif

INSTANTIATE_CLASS(IntraClassCenterLossLayer);
REGISTER_LAYER_CLASS(IntraClassCenterLoss);
}  // namespace caffe

