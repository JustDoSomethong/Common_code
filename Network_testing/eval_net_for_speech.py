#coding=utf-8
# eval_net.py           # 实时捕获摄像头输入,每帧图像送入分类网络,输出

import argparse
import time
import numpy as np
import sys
import cv2
sys.path.append('../Speech_processing/')
from Speech_Class import Thread_for_speech

parser = argparse.ArgumentParser()
parser.add_argument("--caffe_path", type=str, default='/home/pro/caffe-action', help='path to the caffe toolbox')   # caffe:运行位置
parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)    # gpu_id，不用更改
parser.add_argument('--net_proto', type=str,
                    default='/home/pro/WORK/action-recognition/own/BN-Inception/tsn_bn_inception_deploy.prototxt')    # deploy.prototxt 模型
parser.add_argument('--net_weights', type=str,
                    default='/home/pro/WORK/action-recognition/own/BN-Inception/output/_iter_15000.caffemodel')     # caffemodel 模型参数
parser.add_argument('--meanfile_path',type=str,default='/home/pro/WORK/action-recognition/own/imagenet_mean.binaryproto')       # 均值文件
args = parser.parse_args()
print args

sys.path.insert(0, args.caffe_path + '/python')
import caffe

Category_labels =['吃东西',
                  '喝水',
                  '打电话',
                  '看手机',
                  '看书',
                  '头疼',
                  '牙疼',
                  '胸疼',
                  '腰疼',
                  '腹痛',
                  '跌倒',
                  '其他',
                  ]
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
# 捕捉摄像头图像,程序入口
def CaptureVideo():
    cap = cv2.VideoCapture(0)
    res_old = [0,0,0,0,0,0,0,0,0,0,0,1]     # 其他类别
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        start = time.time()
        # Display the resulting frame
        res = eval_video(net_,frame,res_old)    # 更新类别信息
        res_old = res   # 更新类别信息
        cv2.putText(frame, Category_labels_[np.argmax(res)], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        cv2.imshow('Webcam', frame)

        end = time.time()
        seconds = end - start
        # Calculate frames per second
        fps = 1 / seconds
        print("fps: {0}".format(fps))
        # Wait to press 'q' key for break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return

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
# 根据图像帧,保存输出结果
def eval_video(net,frame,res_old):
    # print frame.shape
    global frame_list,Speech_list,Speech_iter,Speech_iter_num,head_Serious_lable,chest_Serious_lable
    net.blobs['data'].data[...] = net._transformer.preprocess('data', frame)  # 执行预处理操作，并将图片载入到blob中
    out = net.forward()
    prob = out['classifiernew']
    if len(frame_list) < Speech_iter:
        frame_list.append(prob)
        res = res_old
    else:
        res = np.mean(frame_list,axis=1).mean(axis=0)
        print Category_labels[np.argmax(res)],np.max(res)
        frame_list = []
        Speech_list.append(np.argmax(res))
        Speech_iter_num += 1
        # print Speech_list
        if len(Speech_list) == Speech_iter:
            Speech_list.pop(0)
            if Speech_iter_num >= Speech_iter:
                if Speech_list.count(7) >= int(0.8*Speech_iter) and chest_Serious_lable == 0:    # 轻微胸疼
                    Thread_for_speech(0)
                    chest_Serious_lable = 1
                    Speech_iter_num = 0
                elif Speech_list.count(7) >= int(0.8*Speech_iter) and chest_Serious_lable == 1:    # 严重胸疼
                    Thread_for_speech(1)
                    chest_Serious_lable = 0
                    Speech_iter_num = 0
                elif Speech_list.count(5) >= int(0.8*Speech_iter) and head_Serious_lable == 0:    # 轻微头疼
                    Thread_for_speech(2)
                    head_Serious_lable = 1
                    Speech_iter_num = 0
                elif Speech_list.count(5) >= int(0.8*Speech_iter) and head_Serious_lable == 1:    # 严重头疼
                    Thread_for_speech(3)
                    head_Serious_lable = 0
                    Speech_iter_num = 0
                elif Speech_list.count(3) >= int(0.8*Speech_iter):  # 看手机
                    Thread_for_speech(4)
                    Speech_iter_num = 0
                elif Speech_list.count(9) >= int(0.8*Speech_iter):  # 腹痛
                    Thread_for_speech(5)
                    Speech_iter_num = 0
                elif Speech_list.count(6) >= int(0.8*Speech_iter):  # 牙疼
                    Thread_for_speech(6)
                    Speech_iter_num = 0

    return res
    # print prob
    # print Category_labels[np.argmax(prob)],np.max(prob)

##############################################################################
# 主函数
if __name__ =='__main__':
    frame_list = []      # 每十帧的输出结果平均
    Speech_list = []     # 连续多帧的识别结果,用于语音播报判断
    Speech_iter = 10     # 每次播放语音间隔
    Speech_iter_num = 0  # 当前语音播报结束后,当前播报间隔
    head_Serious_lable = 0    # 该标签表示疼痛的严重程度(头疼)
    chest_Serious_lable = 0    # 该标签表示疼痛的严重程度(胸疼)
    build_net()     # 网络初始化
    CaptureVideo()      # 实时捕获摄像头,调用分类网络实时输出结果
