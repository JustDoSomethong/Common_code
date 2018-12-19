#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Plot_ROC.py   # 绘制ROC曲线

import argparse
import numpy as np
import sys
from sklearn import metrics
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--caffe_path", type=str, default='/home/pro/caffe-action', help='path to the caffe toolbox')   # caffe:运行位置
parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)    # gpu_id，不用更改
parser.add_argument('--net_proto', type=str,
                    default='/home/pro/WORK/action-recognition/own/BN-Inception/tsn_bn_inception_deploy.prototxt')    # deploy.prototxt 模型
parser.add_argument('--net_weights', type=str,
                    default='/home/pro/WORK/action-recognition/own/BN-Inception/output/_iter_15000.caffemodel')     # caffemodel 模型参数
parser.add_argument('--meanfile_path',type=str,default='/home/pro/WORK/action-recognition/own/imagenet_mean.binaryproto')       # 均值文件
parser.add_argument('--val_file',type=str,default='/media/pro/PRO/Ubuntu/Common code/Data_preprocessing/train_val/val/val.txt')       # 验证集文件
parser.add_argument('--image_path',type=str,default='/media/pro/PRO/Ubuntu/Common code/Data_preprocessing')     # 图像存放路径
parser.add_argument('--classification_label', help='label:0~11', default=11, type=int)    # 对某一类分类标签绘制ROC曲线
args = parser.parse_args()
print args

sys.path.insert(0, args.caffe_path + '/python')
import caffe

##############################################################################
# 网络初始化
def build_net():    # 网络初始化
    global net_
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net_ = caffe.Net(args.net_proto, args.net_weights, caffe.TEST)
  
    input_shape = net_.blobs['data'].data.shape     # 输入图像尺寸
    transformer = caffe.io.Transformer({'data': input_shape})   # 设定维度格式(Num, Channels, Height, Width)
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension

    blob = caffe.proto.caffe_pb2.BlobProto()
    mean = open(args.meanfile_path, 'rb').read()
    blob.ParseFromString(mean)
    data = np.array(caffe.io.blobproto_to_array(blob))
    mean_npy = data[0]
    transformer.set_mean('data', mean_npy.mean(1).mean(1))  # subtract the dataset-mean value in each channel

    # 注意:这两步很重要:
    # 如果是cv2.imread()读取图像,则不要以下两个步骤;
    # 如果是caffe.io.load_image()读取图像,这两个步骤就很有必要;
    transformer.set_raw_scale('data', 255)  # 缩放到[0，255],主要是由于caffe.io.load_image读取图像为float:0-1,RGB图像
    transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，RGB->BGR,需要转化为BGR图像

    net_._transformer = transformer     # 定义网络net_成员
    net_._sample_shape = net_.blobs['data'].data.shape  # 定义网络net_成员
  
##############################################################################
# 计算该类别的准确率,以及伪真类率/真正类率
def compute_fpr_tpr(net):
    right_num = 0
    img_num = 0
    f = open(args.val_file,'r')
    lines = f.readlines()
    feat = [[0 for i in range(3)] for j in range(len(lines))]
    for i,line in enumerate(lines):     # 同时列出数据和数据下标
        line = line.split(' ')
        im = caffe.io.load_image(args.image_path + '/' + str(line[0]))   # float:0-1,RGB图像
        # im = cv2.imread(args.image_path + '/' + line[0])  # float:0-255,BGR图像
        # cv2.imshow("读取视频", im); # 显示当图像
        # cv2.waitKey(30); # 延时30ms
        net.blobs['data'].data[...] = net._transformer.preprocess('data', im)  # 执行预处理操作，并将图片载入到blob中
        out = net.forward()['classifiernew'].flatten()
        print args.image_path + '/' + line[0],np.argmax(out),out[args.classification_label],1-out[args.classification_label]
        feat[i][0] = out[args.classification_label]     # 对应标签的概率
        feat[i][1] = sum(out) - out[args.classification_label]     # 其他类别的概率
        if int(line[1]) == args.classification_label:
            feat[i][2] = 0
            if feat[i][0] > feat[i][1]:
                right_num = right_num + 1
        else:
            feat[i][2] = 1
            if feat[i][1] > feat[i][0]:
                right_num = right_num + 1
        img_num = img_num + 1
        print right_num,img_num

    accuracy = float(right_num)/float(img_num)
    print 'The accuracy of class %d '%args.classification_label,'is %.2f%%,'% (accuracy*100),'the specific numbers are %d / '%right_num,'%d'%img_num
    y_true = [ x[2] for x in feat ] # label,属于该类为0,不属于该类为1
    y_scores = [ x[1] for x in feat ] # caffe输出的图片属于每一个类别的概率

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)

    return fpr, tpr
##############################################################################
# 绘制ROC曲线
def draw_roc(fpr,tpr):
    AUC = metrics.auc(fpr, tpr)
    print 'The Area Under roc Curve(AUC) of class %d'%args.classification_label,'is %.2f'%AUC
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

##############################################################################

if __name__ == '__main__':
    build_net()
    fpr, tpr = compute_fpr_tpr(net_)
    draw_roc(fpr, tpr)
