#!/usr/bin/env python
# coding=utf-8
# eval_detection_classification_speech_net.py      # 检测+分类+语音

import argparse
import time
import numpy as np
import sys
import cv2
import os
sys.path.append('../Speech_processing/')
from Speech_Class import Thread_for_speech

parser = argparse.ArgumentParser()
parser.add_argument("--caffe_path", type=str, default='/home/pro/caffe-ssd', help='path to the caffe toolbox')   # caffe:运行位置
parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)    # gpu_id，不用更改
parser.add_argument('--net_proto_detection', type=str,
                    default='/home/pro/WORK/action-recognition/own/MobileNet-SSD/MobileNetSSD_deploy.prototxt')    # deploy.prototxt 模型
parser.add_argument('--net_weights_detection', type=str,
                    default='/home/pro/WORK/action-recognition/own/MobileNet-SSD/MobileNetSSD_deploy.caffemodel')     # caffemodel 模型参数
parser.add_argument('--net_proto_classification', type=str,
                    default='/media/pro/PRO/YH/UESTC/研究型學習/课题5_行为识别/behavior_recognition/TX2_models/Seventh_LMDB_output/SE-BN-Inception.prototxt')    # deploy.prototxt 模型
parser.add_argument('--net_weights_classification', type=str,
                    default='/media/pro/PRO/YH/UESTC/研究型學習/课题5_行为识别/behavior_recognition/TX2_models/Seventh_LMDB_output/Seventh_LMDB_output.caffemodel')     # caffemodel 模型参数
parser.add_argument('--meanfile_path',type=str,default='/media/pro/PRO/YH/UESTC/研究型學習/课题5_行为识别/behavior_recognition/第七份数据/lmdb/imagenet_mean.binaryproto')       # 均值文件
parser.add_argument('--storage_path',type=str,default='/media/pro/PRO/storage')       # 存储图像路径
args = parser.parse_args()
print args

sys.path.insert(0, args.caffe_path + '/python')
import caffe

Category_labels_ ={ 0:'eating',
                    1:'drinking',
                    2:'calling',
                    3:'reading',
                    4:'reading',
                    5:'headache',
                    6:'toothache',
                    7:'chest pain',
                    8:'waist pain',
                    9:'stomachache',
                    10:'falling',
                    11:'others',
                   }
CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

##############################################################################
# 捕捉摄像头图像,程序入口
def CaptureVideo():
    cap = cv2.VideoCapture(0)
    res_old = [0,0,0,0,0,0,0,0,0,0,0,1]

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        start = time.time()
        # Display the resulting frame
        box, conf, cls = eval_video_detection(net_detection_,frame)

        p_box = []
        for i in range(len(box)):
            p_box_per_person = {}
            if int(cls[i]) == 15 and conf[i] > 0.0:
                p_box_per_person['p1'] = (box[i][0], box[i][1])
                p_box_per_person['p2'] = (box[i][2], box[i][3])
                p_box_per_person['height_difference'] = box[i][3] - box[i][1]
                p_box_per_person['weight_difference'] = box[i][2] - box[i][0]
                p_box_per_person['conf'] = conf[i]
                p_box_per_person['cls'] = int(cls[i])
                p_box.append(p_box_per_person)

        # print p_box
        if len(p_box) == 0:
            cv2.putText(frame, Category_labels_[11], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            cv2.imshow('Webcam', frame)
        else:
            p_box = sorted(p_box, key = lambda e: e.__getitem__('weight_difference'),reverse=True)
            p_hh_low = float(p_box[0]['p1'][1]) - (1 / float((p_box[0]['p2'][1] - p_box[0]['p1'][1]))) * 2000.0  # 记录剪裁框的位置大小
            p_hh_hige = float(p_box[0]['p2'][1]) + (1 / float((p_box[0]['p2'][1] - p_box[0]['p1'][1]))) * 2000.0
            print p_box[0]['p2'][0] - p_box[0]['p1'][0]
            if p_box[0]['p2'][0] - p_box[0]['p1'][0] > 400:
                Penalty_ratio_factor = 30000.0
            elif p_box[0]['p2'][0] - p_box[0]['p1'][0] > 300:
                Penalty_ratio_factor = 15000.0
            elif p_box[0]['p2'][0] - p_box[0]['p1'][0] > 200:
                Penalty_ratio_factor = 8000.0
            elif p_box[0]['p2'][0] - p_box[0]['p1'][0] > 100:
                Penalty_ratio_factor = 3000.0
            else:
                Penalty_ratio_factor = 1000.0
            p_ww_low = float(p_box[0]['p1'][0]) - (1 / float(
                (p_box[0]['p2'][0] - p_box[0]['p1'][0]))) * Penalty_ratio_factor
            p_ww_hige = float(p_box[0]['p2'][0]) + (1 / float(
                (p_box[0]['p2'][0] - p_box[0]['p1'][0]))) * Penalty_ratio_factor
            cropframe = frame[int(max(p_hh_low,0)): int(min(p_hh_hige,frame.shape[0])),int(max(p_ww_low, 0)):int(min(p_ww_hige, frame.shape[1]))]
            res = eval_video_classification(net_classification_, cropframe, res_old)

            cv2.imshow('CropWebcam', cropframe)
            cv2.rectangle(frame, p_box[0]['p1'], p_box[0]['p2'], (0, 0, 255))
            p3 = (max(p_box[0]['p1'][0], 15), max(p_box[0]['p1'][1], 15))
            title = "%s:%.2f" % (CLASSES[int(15)], conf[i])
            cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)

            print res
            res_old = res
            #print np.max(res)
            if np.max(res) < 0.9:
                cv2.putText(frame, Category_labels_[11], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            else:
                Category_res = np.argmax(res)
                Category_res = (Category_res > 11 and 11 or Category_res)
                # print Category_res
                cv2.putText(frame, Category_labels_[Category_res], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
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
    global net_detection_
    global net_classification_
    caffe.set_mode_gpu()        # gpu 设置
    caffe.set_device(args.gpu_id)       # gpu_id 设置
    net_detection_ = caffe.Net(args.net_proto_detection, args.net_weights_detection, caffe.TEST)      # 网络初始化

    net_classification_ = caffe.Net(args.net_proto_classification, args.net_weights_classification, caffe.TEST)      # 网络初始化
    input_shape = net_classification_.blobs['data'].data.shape     # 输入图像尺寸
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

    net_classification_._transformer = transformer     # 定义网络net_成员
    net_classification_._sample_shape = net_classification_.blobs['data'].data.shape  # 定义网络net_成员

##############################################################################
# 获得检测结果输出
def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

##############################################################################
# 检测捕获的帧级图像
def eval_video_detection(net,frame):
    img = cv2.resize(frame, (300, 300))       # resize 300 * 300
    img = img - 127.5
    img = img * 0.007843    # 取均值
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] =  img          # 将图片载入到blob中

    out = net.forward()
    box, conf, cls = postprocess(frame, out)
    return box, conf, cls

##############################################################################
# 对检测的帧级图像分类
def eval_video_classification(net,frame,res_old):
    # print frame.shape
    global frame_list,Speech_list,Speech_iter,Speech_iter_num,head_Serious_lable,chest_Serious_lable
    global iter_
    net.blobs['data'].data[...] = net._transformer.preprocess('data', frame)  # 执行预处理操作，并将图片载入到blob中
    out = net.forward()
    prob = out['classifiernew']
    if len(frame_list) < Speech_iter:
        frame_list.append(prob)
        res = res_old
    else:
        res = np.mean(frame_list,axis=1).mean(axis=0)
        frame_list = []
        Speech_list.append(np.argmax(res))
        Speech_iter_num += 1
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
    if iter_% Speech_iter == 0:
        # print os.path.join(args.storage_path,'img_%08d_%08d.jpg'%(iter_,time.time()))
        cv2.imwrite(os.path.join(args.storage_path,otherStyleTime,'img_%08d_%s.jpg'%(iter_,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime( int(time.time()))))),frame)
    iter_ += 1

    return res

##############################################################################
if __name__ =='__main__':
    frame_list = [] # 每十帧的输出结果平均
    Speech_list = []     # 连续多帧的识别结果,用于语音播报判断
    Speech_iter = 10     # 每次播放语音间隔
    Speech_iter_num = 0  # 当前语音播报结束后,当前播报间隔
    head_Serious_lable = 0    # 该标签表示疼痛的严重程度(头疼)
    chest_Serious_lable = 0    # 该标签表示疼痛的严重程度(胸疼)
    iter_ = 0       # 为了实时保存测试图像
    timeStamp = int(time.time())
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    if not os.path.exists(os.path.join(args.storage_path,otherStyleTime)):
        os.makedirs(os.path.join(args.storage_path,otherStyleTime))
    build_net()     # 网络初始化
    CaptureVideo()      # 实时捕获摄像头,调用分类网络实时输出结果
