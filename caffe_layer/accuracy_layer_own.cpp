#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);	// outer_num_为图像数量，100
  inner_num_ = bottom[0]->count(label_axis_ + 1);	// inner_num_为每个图像所对应的类别数，1
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  int dim = bottom[0]->count() / outer_num_;	// dim = 10
  top[0]->Reshape(1 + dim, 1, 1, 1);
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;		// 准确率初始化为0
  const Dtype* bottom_data = bottom[0]->cpu_data();	// 输入图像100张,每一张对应10个输出类别 100*10
  const Dtype* bottom_label = bottom[1]->cpu_data();	// 图像标签,每一张图像对应一个标签 100*1
  int num = outer_num_;	// 图像总数:100
  const int dim = bottom[0]->count() / outer_num_;	// dim = 10,outer_num_ = 100
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  vector<Dtype> accuracies(dim, 0);	// 记录每个类别的准确率
  vector<Dtype> nums(dim, 0);		// 记录每个类别图像的总数
  for (int i = 0; i < outer_num_; ++i) {
      const int label_value = static_cast<int>(bottom_label[i]);		// 每张图像的标签
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < dim; ++k) {
        bottom_data_vector.push_back(std::make_pair(	// 记录预测结果:dim = 10;inner_num = 1,num_labels = 10
            bottom_data[i * dim + k], k));
      }
      std::partial_sort(	// 按预测结果排序
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {	// 只看前top_k个结果
        ++nums[label_value];
        if (bottom_data_vector[k].second == label_value) {	// 如果存在标签,即准确值增加
          ++accuracy;
          ++accuracies[label_value];	// 对应每个类别准确率计数 + 1
          break;
        }
      }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / num;	// 总的准确率
  for (int i = 0; i < dim; ++i) {		// 对应每个类别的准确率
     top[0]->mutable_cpu_data()[i + 1] = accuracies[i] / nums[i];	// 输出每个类别的准确率
  }
  // Accuracy layer should not be used as a loss function.
}
INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
