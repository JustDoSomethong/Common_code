#ifndef CAFFE_CENTER_AND_INTER_CLASS_LOSS_LAYER_HPP_
#define CAFFE_CENTER_AND_INTER_CLASS_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class CenterAndInterClassLossLayer : public LossLayer<Dtype> {
 public:
  explicit CenterAndInterClassLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CenterAndInterClassLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;   // mini_batch
  int K_;   // 特征输入的长度
  int N_;   // 输出神经元的个数
  
  Blob<Dtype> distance_;
  Blob<Dtype> variation_sum_;    // 记录样本距离对应类中心的距离
  Blob<Dtype> inter_class_diff_;    //记录类间距离,类内距离
  Blob<Dtype> inter_class_dist_sq_;    //记录类间距离平方
};

}  // namespace caffe

#endif  // CAFFE_CENTER_AND_INTER_CLASS_LOSS_LAYER_HPP_