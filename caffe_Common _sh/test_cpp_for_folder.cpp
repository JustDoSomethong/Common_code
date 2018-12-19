#include <caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;
using namespace cv;
using namespace std;

int main(int argc, char** argv){
  // 定义输出文件,模型配置文件,模型文件,均值文件以及测试图像
  std::ofstream out("/home/pro/Project/NanRi/inception_elec_open_close/folder_test /test_inception_first_.txt");
  string model_file   = "/home/pro/Project/NanRi/inception_elec_open_close/folder_test /deploy_inception_v4.prototxt";
  string trained_file = "/home/pro/Project/NanRi/inception_elec_open_close/folder_test /_iter_50000.caffemodel";
  string mean_file    = "/home/pro/Project/NanRi/inception_elec_open_close/folder_test /imagenet_mean.binaryproto";
  string dir_path     = "/home/pro/Project/NanRi/inception_elec_open_close/folder_test /picture/";
  string fin_res[12]  = {"鸟巢","树木","违建","导线覆冰","锈蚀","绝缘子闪络","输电线路场景","隔离开关","开关分合","表计检测","变压器","变电站场景"};
  // 定义变量
  shared_ptr<Net<float> > net_;		// 保存模型
  cv::Size input_geometry_;		// 模型输入图像的尺寸
  int num_channels_;			// 图像的通道数
  cv::Mat mean_;			// 根据均值文件计算得到的均值文件

  Caffe::set_mode(Caffe::GPU);		// 使用GPU
  net_.reset(new Net<float>(model_file, caffe::TEST));		// 加载配置文件
  net_->CopyTrainedLayersFromBinaryProto(trained_file);		// 加载训练好的模型修改模型参数
	
  Blob<float>* input_layer = net_->input_blobs()[0];		// 定义输入层变量
  num_channels_ = input_layer->channels();			// 得到输入层的通道数
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());		// 得到输入层图像的大小

  // 处理均值文件,得到均值图像
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }
  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);		//得到均值图像  

 
  cv::Mat img,sample,sample_resized,sample_float,sample_normalized;
  Blob<float>* output_layer;
  const float* prob;

  cv::Directory dir;
  vector<string> fileNames = dir.GetListFiles(dir_path, ".");
  for (int i = 0; i < fileNames.size(); i++)
  {
  string img_file = dir_path + fileNames[i];
  std::cout << "Prediction for "<< fileNames[i] << " ----------" << std::endl;
  out << "Prediction for "<< fileNames[i] << ": ";
  
  // reshape模型的输入尺寸
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  net_->Reshape();		// 调整模型的大小
  std::vector<cv::Mat> input_channels;  
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels.push_back(channel);
    input_data += width * height;
  }
  // 改变图像的通道数,resize图像的大小
  img = imread(img_file);
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;
  // change img size 
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;
  // change ing to float
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);
  // img normalize 
  cv::subtract(sample_float, mean_, sample_normalized);
  // 将图像通过input_channels变量传递给模型
  cv::split(sample_normalized, input_channels);

  net_->Forward();		// 调整模型进行预测
  output_layer = net_->output_blobs()[0];
  prob = output_layer->cpu_data();
  std::cout << "多标签分类结果: ";
  for(int i =0; i < 12; i++) {
  std::cout << (prob[i]>0 ? 1 : 0) << " ";
  out << (prob[i]>0 ? 1 : 0) << " ";
  }
  out << "\n多标签分类结果为:\t";
  if(prob[6] > 0) out << fin_res[6]<< ": ";
  else if(prob[11] > 0) out << fin_res[11] << ": ";
  else out << "为正确区分输电线路场景和变电站场景!! ";
  for(int i =0; i < 12; i++){
     if (i!=6&&i!=11){
	if(prob[i]>0) out << fin_res[i] << "; ";  
     }
  }
  std::cout << std::endl;
  out << std::endl;
  }
  out.close();
  return 1;
}
