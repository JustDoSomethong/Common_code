#coding=utf-8
# Split_image_from_detection	# 根据检测框分割图像

import argparse
import sys
from Creat_Folders import Creat_Folders_
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--caffe_path", type=str, default='/home/pro/caffe-ssd', help='path to the caffe toolbox')   # caffe:运行位置
parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)    # gpu_id，不用更改
parser.add_argument('--net_proto_detection', type=str,
                    default='/home/pro/WORK/action-recognition/own/MobileNet-SSD/MobileNetSSD_deploy.prototxt')    # deploy.prototxt 模型
parser.add_argument('--net_weights_detection', type=str,
                    default='/home/pro/WORK/action-recognition/own/MobileNet-SSD/MobileNetSSD_deploy.caffemodel')     # caffemodel 模型参数
parser.add_argument('--input_dir',type=str,default='./最终数据整理',help='输入文件夹位置')
parser.add_argument('--output_dir',type=str,default='./检测数据结果',help='输出文件夹保存位置')
args = parser.parse_args()
print args

sys.path.insert(0, args.caffe_path + '/python')
import caffe

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

    net.blobs['data'].data[...] = img          # 将图片载入到blob中
    out = net.forward()
    box, conf, cls = postprocess(frame, out)
    return box, conf, cls

##############################################################################
# 加载图像
def load_picture(input_dir ,image_format = 'jpg'):
    picture_list = []   # 用来存储文件夹下文件
    imageset = os.listdir(input_dir)
    for imagename in imageset:
        if imagename.endswith(image_format):
            picture_list.append(imagename)
    return picture_list

##############################################################################
# 检测图像,并保存结果:
def eval_folder(net,picture_list,input_dir,output_dir):
    for i in range(len(picture_list)):
        # for k in range(100,9,-10):
        k = 10000
        image = cv2.imread(os.path.join(input_dir, picture_list[i]))
        box, conf, cls = eval_video(net,image)
        p_box = []
        for j in range(len(box)):
            p_box_per_person = {}
            if int(cls[j]) == 15 and conf[j] > 0.5:
                p_box_per_person['p1'] = (box[j][0], box[j][1])
                p_box_per_person['p2'] = (box[j][2], box[j][3])
                p_box_per_person['height_difference'] = box[j][3] - box[j][1]
                p_box_per_person['weight_difference'] = box[j][2] - box[j][0]
                p_box_per_person['conf'] = conf[j]
                p_box_per_person['cls'] = int(cls[j])
                p_box.append(p_box_per_person)

        if len(p_box) == 0:
            cv2.imwrite(output_dir + '/%05d.jpg' % i, image)
            # break
        else:
            p_box = sorted(p_box, key=lambda e: e.__getitem__('weight_difference'), reverse=True)
            p_hh_low = int(float(p_box[0]['p1'][1]) - float((p_box[0]['p2'][1] - p_box[0]['p1'][1])) / float(k)) # 记录剪裁框的位置大小
            p_hh_hige = int(float(p_box[0]['p2'][1]) + float((p_box[0]['p2'][1] - p_box[0]['p1'][1])) / float(k))
            p_ww_low = int(float(p_box[0]['p1'][0]) - float((p_box[0]['p2'][0] - p_box[0]['p1'][0])) / float(k))
            p_ww_hige = int(float(p_box[0]['p2'][0]) + float((p_box[0]['p2'][0] - p_box[0]['p1'][0])) / float(k))
            # cv2.rectangle(image, p_box[0]['p1'], p_box[0]['p2'], (0, 255, 0))
            # cv2.rectangle(image, (int(max(p_ww_low, 0)), int(max(p_hh_low, 0))),
            #               (int(min(p_ww_hige, image.shape[1])), int(min(p_hh_hige, image.shape[0]))), (0, 0, 255))
            cropframe = image[int(max(p_hh_low, 0)): int(min(p_hh_hige, image.shape[0])),
                        int(max(p_ww_low, 0)):int(min(p_ww_hige, image.shape[1]))]
            # wid = int(max(p_hh_hige-p_hh_low,p_ww_hige-p_ww_low)/2)
            # xmid = (p_ww_hige + p_ww_low)/2
            # ymid = (p_hh_hige + p_hh_low)/2
            # p_ww_low = xmid - wid
            # p_ww_hige = xmid + wid
            # p_hh_low = ymid - wid
            # p_hh_hige = ymid + wid
            # if p_hh_low < 0 or p_hh_hige > image.shape[0] or p_ww_low > 0 or p_ww_hige > image.shape[1]:
            #     break
            # cropframe = image[int(max(p_hh_low, 0)): int(min(p_hh_hige, image.shape[0])),
            #             int(max(p_ww_low, 0)):int(min(p_ww_hige, image.shape[1]))]
            cv2.imwrite(output_dir + '/%05d_%d.jpg' % (i,k), cropframe)

if __name__ == '__main__':
    out_dir_ = Creat_Folders_(args.input_dir, args.output_dir)  # 根据输入格式,创建输出文件夹
    build_net()
    folder_dir = os.listdir(args.input_dir)
    for image_dir in folder_dir:
        print "Split image for %s" % image_dir
        if image_dir != "其他":
            continue
        image_out_dir = os.path.join(args.output_dir, image_dir)
        image_dir = os.path.join(args.input_dir,image_dir)
        picture_list = load_picture(image_dir)
        eval_folder(net_detection_,picture_list,image_dir,image_out_dir)
        print "Done %s" % image_out_dir

