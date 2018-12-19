#coding=utf-8
# eval_detection_classification_net.py      # 实时捕获摄像头输入,每帧图像送入检测网络,输出人检测结果,随后送入分类网络,输出

import argparse
import time
import numpy as np
import sys
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--caffe_path", type=str, default='/home/pro/caffe-ssd', help='path to the caffe toolbox')   # caffe:运行位置
parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)    # gpu_id，不用更改
parser.add_argument('--net_proto_detection', type=str,
                    default='/home/pro/WORK/action-recognition/own/MobileNet-SSD/example/MobileNetSSD_deploy.prototxt')    # deploy.prototxt 模型
parser.add_argument('--net_weights_detection', type=str,
                    default='/home/pro/WORK/action-recognition/own/MobileNet-SSD/example/mobilenet_iter_117000_deploy.caffemodel')     # caffemodel 模型参数
# parser.add_argument('--meanfile_path',type=str,default='/home/pro/WORK/action-recognition/own/imagenet_mean.binaryproto')       # 均值文件
args = parser.parse_args()
print args

sys.path.insert(0, args.caffe_path + '/python')
import caffe

#CLASSES = ('background',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('background',
           'person')

##############################################################################
# 捕捉摄像头图像,程序入口
def CaptureVideo():
    cap = cv2.VideoCapture(0)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        start = time.time()
        # Display the resulting frame
        box, conf, cls = eval_video(net_detection_,frame)

        p_box = []
        for i in range(len(box)):
            p_box_per_person = {}
            if int(cls[i]) == 1 and conf[i] > 0.5:
                p_box_per_person['p1'] = (box[i][0], box[i][1])
                p_box_per_person['p2'] = (box[i][2], box[i][3])
                p_box_per_person['height_difference'] = box[i][3] - box[i][1]
                p_box_per_person['weight_difference'] = box[i][2] - box[i][0]
                p_box_per_person['conf'] = conf[i]
                p_box_per_person['cls'] = int(cls[i])
                p_box.append(p_box_per_person)

        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(frame, p1, p2, (0,255,0))
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)

        cv2.imshow("SSD", frame)
        #if len(p_box) == 0:
        #    cv2.imshow('Webcam', frame)
        #else:
        #    p_box = sorted(p_box, key = lambda e: e.__getitem__('weight_difference'),reverse=True)
        #    p_hh_low = float(p_box[0]['p1'][1]) - float((p_box[0]['p2'][1]-p_box[0]['p1'][1]))/5.0      # 记录剪裁框的位置大小
        #    p_hh_hige = float(p_box[0]['p2'][1]) + float((p_box[0]['p2'][1]-p_box[0]['p1'][1]))/5.0
        #    p_ww_low = float(p_box[0]['p1'][0]) - float((p_box[0]['p2'][0]-p_box[0]['p1'][0]))/5.0
        #    p_ww_hige = float(p_box[0]['p2'][0]) + float((p_box[0]['p2'][0]-p_box[0]['p1'][0]))/5.0
        #    cv2.rectangle(frame, p_box[0]['p1'], p_box[0]['p2'], (0, 255, 0))
        #    cv2.rectangle(frame, (int(max(p_ww_low, 0)),int(max(p_hh_low,0))), (int(min(p_ww_hige, frame.shape[1])),int(min(p_hh_hige,frame.shape[0]))), (0, 0, 255))
        #    p3 = (max(p_box[0]['p1'][0], 15), max(p_box[0]['p1'][1], 15))
        #    title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        #    cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
        #    cv2.imshow('Webcam', frame)   

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
    caffe.set_mode_gpu()        # gpu 设置
    caffe.set_device(args.gpu_id)       # gpu_id 设置
    net_detection_ = caffe.Net(args.net_proto_detection, args.net_weights_detection, caffe.TEST)      # 网络初始化

    # input_shape = net_.blobs['data'].data.shape     # 输入图像尺寸
    # transformer = caffe.io.Transformer({'data': input_shape})   # 设定维度格式(Num, Channels, Height, Width)
    # transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    #
    # blob = caffe.proto.caffe_pb2.BlobProto()
    # mean = open(args.meanfile_path, 'rb').read()
    # blob.ParseFromString(mean)
    # data = np.array(caffe.io.blobproto_to_array(blob))
    # mean_npy = data[0]
    # transformer.set_mean('data', mean_npy.mean(1).mean(1))  # subtract the dataset-mean value in each channel
    #
    # # # 注意:这两步很重要:
    # # # 如果是cv2.imread()读取图像,则不要以下两个步骤;
    # # # 如果是caffe.io.load_image()读取图像,这两个步骤就很有必要;
    # # transformer.set_raw_scale('data', 255)  # 缩放到[0，255],主要是由于caffe.io.load_image读取图像为float:0-1,RGB图像
    # # transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，RGB->BGR,需要转化为BGR图像
    #
    # net_._transformer = transformer     # 定义网络net_成员
    # net_._sample_shape = net_.blobs['data'].data.shape  # 定义网络net_成员

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
def eval_video(net,frame):

    global frame_list
    img = cv2.resize(frame, (300, 300))       # resize 300 * 300
    img = img - 127.5
    img = img * 0.007843    # 取均值
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] =  img          # 将图片载入到blob中

    out = net.forward()
    box, conf, cls = postprocess(frame, out)
    return box, conf, cls

if __name__ =='__main__':
    frame_list = []
    build_net()     # 网络初始化
    CaptureVideo()      # 实时捕获摄像头,调用分类网络实时输出结果
