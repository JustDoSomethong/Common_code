caffe常用sh/Train_Val.py		制作 train.txt/test.txt 清单文件
caffe常用sh/covert.sh			文件resize图片,改变图片的大小
caffe常用sh/create_lmdb.sh		文件产生lmdb文件,输入为txt文件,存有图像的路径以及标签(convert_imageset)
caffe常用sh/make_imagenet_mean.sh	文件获取图像的均值,输入为lmdb文件train_lmdb(compute_image_mean)
caffe常用sh/solver.prototxt		文件为训练模型时的参数设置.step设置学习率下降
caffe常用sh/multistep_solver.prototxt	文件为训练模型时的参数设置.multistep设置学习率下降
caffe常用sh/train.sh			文件用caffe训练网络(caffe train)
caffe常用sh/test.sh			文件用caffe测试网络(caffe test)
caffe常用sh/resume_training.sh		文件用于恢复数据进行再训练(caffe train)
caffe常用sh/train_val_txt.m		文件为处理图片,生成相应的train和val数据,是matlab文件
caffe常用sh/VOC_ImageSet_Main_trainvaltest.m		文件为在Faster R-CNN中,仿照VOC数据集的时候,生成对应的train val test 文件.是matlab文件.可以修改为己用.
caffe常用sh/test_python.py	        文件为caffe测试程序,是Python文件
caffe常用sh/test_python_for_picture.py	文件为caffe测试程序,是python文件,是单张图片的测试程序
caffe常用sh/test_cpp.cpp		文件为caffe测试程序,是cpp文件,是单张图片的测试程序
caffe常用sh/test_cpp_for_folder.cpp	文件为caffe测试程序,是cpp文件,是测试文件夹中所有图片
caffe常用sh/Calculate_accuracy.py	文件为py程序,计算准确率
caffe常用sh/fuse_model.py		文件为py程序，融合模型，生成融合后的权重。


caffe常用sh/Show_Pics.py		文件为python程序,可以同时在一个窗口中显示多张图片
caffe常用sh/Picture_filter.py		文件为Windows扒图代码
caffe常用sh/Read_COCO.py		文件为根据COCO_json文件读取COCO图片
caffe常用sh/data_preprocessing.py	文件为腾讯算法大赛数据处理代码,其中包含Python：pandas数据处理,以及sklearn库相关操作
