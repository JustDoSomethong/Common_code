#coding=utf-8
# visualize_deepfeature.py	# 查看center_loss的作用,即ip1层的输出结果

import argparse
import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir',type=str,default='/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/第一份数据/train_val/train/train.txt')
parser.add_argument('--data_dir',type=str,default='/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/第一份数据')
parser.add_argument("--caffe_path", type=str, default='/home/pro/caffe-master', help='path to the caffe toolbox')   # caffe:运行位置
parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)    # gpu_id，不用更改
parser.add_argument('--net_proto', type=str,
                    default='/home/pro/WORK/action-recognition/own/SE-BN-Inception_iter_class_itra_class_center_loss/deploy.prototxt')    # deploy.prototxt 模型
parser.add_argument('--net_weights', type=str,
                    default='/media/pro/PRO/YH/UESTC/研究型學習/课题5_行为识别/behavior_recognition/SE-BN-Inception_iter_class_itra_class_center_loss/output/SE_BN_inter_intra_class_2cishiyan_50000_0.892696.caffemodel')     # caffemodel 模型参数
parser.add_argument('--meanfile_path',type=str,default='/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/第一份数据/lmdb/imagenet_mean.binaryproto')       # 均值文件
args = parser.parse_args()
print args

sys.path.insert(0, args.caffe_path + '/python')
import caffe

Category_labels_ ={ 0:'eating',
                    1:'drinking',
                    2:'calling',
                    3:'read phone',
                    4:'reading',
                    5:'headache',
                    6:'toothache',
                    7:'chest pain',
                    8:'waist pain',
                    9:'stomachache',
                    10:'falling',
                    11:'others',
                   }

##############################################################################
# 网络初始化
def build_net():    # 网络初始化
    global net_
    caffe.set_mode_gpu()        # gpu 设置
    caffe.set_device(args.gpu_id)       # gpu_id 设置
    net_ = caffe.Net(args.net_proto, args.net_weights, caffe.TEST)      # 网络初始化

    input_shape = net_.blobs['data'].data.shape     # 输入图像尺寸
    transformer = caffe.io.Transformer({'data': input_shape})   # 设定维度格式(Num, Channels, Height, Width)
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension

    blob = caffe.proto.caffe_pb2.BlobProto()
    mean = open(args.meanfile_path, 'rb').read()
    blob.ParseFromString(mean)
    data = np.array(caffe.io.blobproto_to_array(blob))
    mean_npy = data[0]
    transformer.set_mean('data', mean_npy.mean(1).mean(1))  # subtract the dataset-mean value in each channel

    # # 注意:这两步很重要:
    # # 如果是cv2.imread()读取图像,则不要以下两个步骤;
    # # 如果是caffe.io.load_image()读取图像,这两个步骤就很有必要;
    # transformer.set_raw_scale('data', 255)  # 缩放到[0，255],主要是由于caffe.io.load_image读取图像为float:0-1,RGB图像
    # transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，RGB->BGR,需要转化为BGR图像

    net_._transformer = transformer     # 定义网络net_成员
    net_._sample_shape = net_.blobs['data'].data.shape  # 定义网络net_成员

##############################################################################
# 加载图像
def load_picture(train_dir = args.train_dir,image_format = 'jpg'):
    picture_list = []   # 用来存储文件夹下文件
    with open(train_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            picture_list.append(line)
    return picture_list

##############################################################################
# 根据图像,保存输出结果
def eval_video(net,data_dir = args.data_dir):
    center_feat_write = open('./center_feat.txt', 'w')
    for i in range(len(picture_list)):
        # print os.path.join(data_dir,picture_list[i].split(' ')[0])
        image = cv2.imread(os.path.join(data_dir,picture_list[i].split(' ')[0]))
        net.blobs['data'].data[...] = net._transformer.preprocess('data', image)  # 执行预处理操作，并将图片载入到blob中
        out = net.forward()
        res_data = out['ip1']
        res = [picture_list[i].strip().split(' ')[1],'%.5f'%res_data[0][0],'%.5f'%res_data[0][1],'\n']
        center_feat_write.write(' '.join(res))
        res = []
    center_feat_write.close()

##############################################################################
# 绘制图像
def draw_deep_feature(path):
    with open(path,'r') as f:
        lines = f.readlines()

    d = dict()

    for line in lines:
        sps = line.split()
        label = int(sps[0])
        if label not in d:
            d[label] = []
        pt = [float(sps[1]), float(sps[2])]
        d[label].append(pt)
    print len(d)
    color = ['blue','green','red','cyan','magenta',
        'yellow','black','deeppink','gray','pink','darksalmon','indigo']
    for key in d:
        print str(key) + "\t" + str(len(d[key]))
        val = np.array(d[key])
        x = val[:,0]
        y = val[:,1]
        plt.scatter(x,y, label=key, c = color[key])

    plt.legend()
    plt.show()
##############################################################################
# 主函数
if __name__ =='__main__':
    build_net()     # 网络初始化
    picture_list = load_picture()     # 加载图像
    eval_video(net_)
    draw_deep_feature('center_feat.txt')
