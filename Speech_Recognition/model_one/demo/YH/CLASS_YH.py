# coding=utf-8
# CLASS_YH.py      # 检测+分类+语音

import argparse
import time
import numpy as np
import sys
import cv2
import os
sys.path.append('./Speech_processing/')
from Speech_Class import Thread_for_speech

#############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--caffe_path", type=str, default='/home/pro/caffe-ssd', help='path to the caffe toolbox')   # caffe:运行位置
parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)    # gpu_id，不用更改
parser.add_argument('--net_proto_detection', type=str,
                    default='./YH/MobileNetSSD_deploy.prototxt')    # deploy.prototxt 模型
parser.add_argument('--net_weights_detection', type=str,
                    default='./YH/MobileNetSSD_deploy.caffemodel')     # caffemodel 模型参数
parser.add_argument('--net_proto_classification', type=str,
                    default='./YH/SE-BN-Inception.prototxt')    # deploy.prototxt 模型
parser.add_argument('--net_weights_classification', type=str,
                    default='./YH/Seventh_LMDB_output.caffemodel')     # caffemodel 模型参数
parser.add_argument('--meanfile_path',type=str,default='./YH/imagenet_mean.binaryproto')       # 均值文件
parser.add_argument('--storage_path',type=str,default='./YH/storage')       # 存储图像路径
args = parser.parse_args()
# print args

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

class YH_detection_classification:
    def init(self):
        self.frame_list = []  # 每十帧的输出结果平均
        self.Speech_list = []     # 连续多帧的识别结果,用于语音播报判断
        self.Speech_iter = 10     # 每次播放语音间隔
        self.Speech_iter_num = 0  # 当前语音播报结束后,当前播报间隔
        self.head_Serious_lable = 0    # 该标签表示疼痛的严重程度(头疼)
        self.chest_Serious_lable = 0    # 该标签表示疼痛的严重程度(胸疼)
        self.iter_ = 0       # 为了实时保存测试图像
        self.res_old = [0,0,0,0,0,0,0,0,0,0,0,1]    # 初始状态为其他

        # 创建保存的文件夹
        timeStamp = int(time.time())
        timeArray = time.localtime(timeStamp)
        self.otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        if not os.path.exists(os.path.join(args.storage_path,self.otherStyleTime)):
            os.makedirs(os.path.join(args.storage_path,self.otherStyleTime))

        caffe.set_mode_gpu()  # gpu 设置
        caffe.set_device(args.gpu_id)  # gpu_id 设置
        self.net_detection_ = caffe.Net(args.net_proto_detection, args.net_weights_detection, caffe.TEST)  # 网络初始化

        self.net_classification_ = caffe.Net(args.net_proto_classification, args.net_weights_classification,
                                        caffe.TEST)  # 网络初始化
        input_shape = self.net_classification_.blobs['data'].data.shape  # 输入图像尺寸
        self.transformer = caffe.io.Transformer({'data': input_shape})  # 设定维度格式(Num, Channels, Height, Width)
        self.transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension

        blob = caffe.proto.caffe_pb2.BlobProto()
        mean = open(args.meanfile_path, 'rb').read()
        blob.ParseFromString(mean)
        data = np.array(caffe.io.blobproto_to_array(blob))
        mean_npy = data[0]
        self.transformer.set_mean('data', mean_npy.mean(1).mean(1))  # subtract the dataset-mean value in each channel

        # # 注意:这两步很重要:
        # # 如果是cv2.imread()读取图像,则不要以下两个步骤;
        # # 如果是caffe.io.load_image()读取图像,这两个步骤就很有必要;
        # transformer.set_raw_scale('data', 255)  # 缩放到[0，255],主要是由于caffe.io.load_image读取图像为float:0-1,RGB图像
        # transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，RGB->BGR,需要转化为BGR图像

        self.net_classification_._transformer = self.transformer  # 定义网络net_成员
        self.net_classification_._sample_shape = self.net_classification_.blobs['data'].data.shape  # 定义网络net_成员

    def forward(self,frame):
        start = time.time()
        # Display the resulting frame
        box, conf, cls = self.eval_video_detection(frame)
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
            # print p_box[0]['p2'][0] - p_box[0]['p1'][0]
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
            res = self.eval_video_classification(cropframe)

            #cv2.imshow('CropWebcam', cropframe)
            cv2.rectangle(frame, p_box[0]['p1'], p_box[0]['p2'], (0, 0, 255))
            p3 = (max(p_box[0]['p1'][0], 15), max(p_box[0]['p1'][1], 15))
            title = "%s:%.2f" % (CLASSES[int(15)], conf[i])
            cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)

            # print res
            self.res_old = res
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
        # cv2.waitKey(1)

    #############################################################################
    # 获得检测结果输出
    def postprocess(self,img,out):
        h = img.shape[0]
        w = img.shape[1]
        box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
        cls = out['detection_out'][0,0,:,1]
        conf = out['detection_out'][0,0,:,2]
        return (box.astype(np.int32), conf, cls)

    ##############################################################################
    # 检测捕获的帧级图像
    def eval_video_detection(self,frame):
        img = cv2.resize(frame, (300, 300))       # resize 300 * 300
        img = img - 127.5
        img = img * 0.007843    # 取均值
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))

        self.net_detection_.blobs['data'].data[...] = img          # 将图片载入到blob中

        out = self.net_detection_.forward()
        box, conf, cls = self.postprocess(frame,out)
        return box, conf, cls

    ##############################################################################
    # 对检测的帧级图像分类
    def eval_video_classification(self,frame):
        # print frame.shape
        # global frame_list,Speech_list,Speech_iter,Speech_iter_num,head_Serious_lable,chest_Serious_lable
        # global iter_
        self.net_classification_.blobs['data'].data[...] = self.net_classification_._transformer.preprocess('data', frame)  # 执行预处理操作，并将图片载入到blob中
        out = self.net_classification_.forward()
        prob = out['classifiernew']
        if len(self.frame_list) < self.Speech_iter:
            self.frame_list.append(prob)
            res = self.res_old
        else:
            res = np.mean(self.frame_list,axis=1).mean(axis=0)
            self.frame_list = []
            self.Speech_list.append(np.argmax(res))
            self.Speech_iter_num += 1
            if len(self.Speech_list) == self.Speech_iter:
                self.Speech_list.pop(0)
                if self.Speech_iter_num >= self.Speech_iter:
                    if self.Speech_list.count(7) >= int(0.8*self.Speech_iter) and self.chest_Serious_lable == 0:    # 轻微胸疼
                        Thread_for_speech(0)
                        self.chest_Serious_lable = 1
                        self.Speech_iter_num = 0
                    elif self.Speech_list.count(7) >= int(0.8*self.Speech_iter) and self.chest_Serious_lable == 1:    # 严重胸疼
                        Thread_for_speech(1)
                        self.chest_Serious_lable = 0
                        self.Speech_iter_num = 0
                    elif self.Speech_list.count(5) >= int(0.8*self.Speech_iter) and self.head_Serious_lable == 0:    # 轻微头疼
                        Thread_for_speech(2)
                        self.head_Serious_lable = 1
                        self.Speech_iter_num = 0
                    elif self.Speech_list.count(5) >= int(0.8*self.Speech_iter) and self.head_Serious_lable == 1:    # 严重头疼
                        Thread_for_speech(3)
                        self.head_Serious_lable = 0
                        self.Speech_iter_num = 0
                    elif self.Speech_list.count(3) >= int(0.8*self.Speech_iter):  # 看手机
                        Thread_for_speech(4)
                        self.Speech_iter_num = 0
                    elif self.Speech_list.count(9) >= int(0.8*self.Speech_iter):  # 腹痛
                        Thread_for_speech(5)
                        self.Speech_iter_num = 0
                    elif self.Speech_list.count(6) >= int(0.8*self.Speech_iter):  # 牙疼
                        Thread_for_speech(6)
                        self.Speech_iter_num = 0
        if self.iter_% self.Speech_iter == 0:
            # print os.path.join(args.storage_path,'img_%08d_%08d.jpg'%(iter_,time.time()))
            cv2.imwrite(os.path.join(args.storage_path,self.otherStyleTime,'img_%08d_%s.jpg'%(self.iter_,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime( int(time.time()))))),frame)
        self.iter_ += 1

        return res

if __name__ == '__main__':
    YH = YH_detection_classification()
    YH.init()
    cap = cv2.VideoCapture(0)
    while(1):
        ret, frame = cap.read()
        YH.forward(frame)
