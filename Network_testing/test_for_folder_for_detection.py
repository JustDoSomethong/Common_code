# coding=utf-8
# test_for_folder_for_detection         # 检测文件下的图像,输出检测结果

import argparse
import sys
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',type=str,default='/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/shiyandata/One_person_data',help='输入文件夹位置')
parser.add_argument('--output_dir',type=str,default='/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/shiyandata/One_person_data_crop',help='输出文件夹保存位置')
parser.add_argument("--store", help="store_lable",action="store_true")
parser.add_argument('--output_txt_dir',type=str,default='/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/SSD_data',help='输出标签文件保存位置')
parser.add_argument("--caffe_path", type=str, default='/home/pro/caffe-ssd', help='path to the caffe toolbox')   # caffe:运行位置
parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)    # gpu_id，不用更改
parser.add_argument('--net_proto_detection', type=str,
                    default='/home/pro/WORK/action-recognition/own/MobileNet-SSD/deploy.prototxt')    # deploy.prototxt 模型
parser.add_argument('--net_weights_detection', type=str,
                    default='/home/pro/WORK/action-recognition/own/MobileNet-SSD/MobileNetSSD_deploy.caffemodel')     # caffemodel 模型参数
# parser.add_argument('--meanfile_path',type=str,default='/home/pro/WORK/action-recognition/own/imagenet_mean.binaryproto')       # 均值文件
args = parser.parse_args()
print args

sys.path.insert(0, args.caffe_path + '/python')
import caffe

CLASSES = ('background',
           'person')

# Category_labels_ ={ '000':'eating',
#                     '001':'drinking',
#                     '002':'calling',
#                     '003':'reading',
#                     '004':'reading',
#                     '005':'headache',
#                     '006':'toothache',
#                     '007':'chest_pain',
#                     '008':'waist_pain',
#                     '009':'stomachache',
#                     '010':'falling',
#                     '011':'others',
#                     '012':'others',
#                    }

##################################################################
# 如果输出文件夹不存在,创建文件夹,
def Creat_Folders_(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

##############################################################################
# 网络初始化
def build_net():    # 网络初始化
    global net_detection_
    caffe.set_mode_gpu()        # gpu 设置
    caffe.set_device(args.gpu_id)       # gpu_id 设置
    net_detection_ = caffe.Net(args.net_proto_detection, args.net_weights_detection, caffe.TEST)  # 网络初始化

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
# 加载图像
def load_picture(input_dir = args.input_dir,image_format = 'jpg'):
    picture_list = []   # 用来存储文件夹下文件
    imageset = os.listdir(input_dir)
    for imagename in imageset:
        if imagename.endswith(image_format):
            picture_list.append(imagename)
    return picture_list

##############################################################################
# 根据图像,保存输出结果
def eval_picture(net,input_dir = args.input_dir,output_dir=args.output_dir,store_lable=args.store,output_txt_dir=args.output_txt_dir):
    # if store_lable:
    #     Creat_Folders_(output_dir)
    # else:
    #     f = open(os.path.join(output_txt_dir,'shared_label.txt'),'w')
    for ins in range(len(picture_list)):
        image = cv2.imread(os.path.join(input_dir,picture_list[ins]))
        # net.blobs['data'].data[...] = net._transformer.preprocess('data', image)  # 执行预处理操作，并将图片载入到blob中
        # out = net.forward()
        # prob = out['classifiernew']
        # if np.argmax(prob)!=int(input_dir.split('/')[-1]):
        #     print "The %s has the wrong result, %d/%d" %(picture_list[i], np.argmax(prob),int(input_dir.split('/')[-1]))
        box, conf, cls = eval_video(net, image)
        p_box = []
        for i in range(len(box)):
            p_box_per_person = {}
            if int(cls[i]) == 15 and conf[i] > 0.5:
                p_box_per_person['p1'] = (box[i][0], box[i][1])
                p_box_per_person['p2'] = (box[i][2], box[i][3])
                p_box_per_person['height_difference'] = box[i][3] - box[i][1]
                p_box_per_person['weight_difference'] = box[i][2] - box[i][0]
                p_box_per_person['conf'] = conf[i]
                p_box_per_person['cls'] = int(cls[i])
                p_box.append(p_box_per_person)

        # if store_lable:
        if len(p_box) != 0:
            # print os.path.join(input_dir,picture_list[ins])
            # print(image.shape)
            p_box = sorted(p_box, key=lambda e: e.__getitem__('weight_difference'), reverse=True)
            r = p_box[0]['weight_difference']*0.3/2
            p1 = (int(max(p_box[0]['p1'][0]-r,0)), int(max(p_box[0]['p1'][1]-r,0)))
            p2 = (int(min(p_box[0]['p2'][0]+r,image.shape[1])), int(min(p_box[0]['p2'][1]+r,image.shape[0])))
            # print(p_box[0]['p1'][0]-r)
            # print(p_box[0]['p1'][1]-r)
            # print(p_box[0]['p2'][0]+r)
            # print(p_box[0]['p2'][1]+r)
            cropframe = image[p1[1]:p2[1],p1[0]:p2[0]]
            # cv2.rectangle(image, p1, p2, (0, 255, 0))
            cv2.imwrite(os.path.join(output_dir,picture_list[ins]), cropframe)
            image_skl = cv2.imread(os.path.join('/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/shiyandata/Skeleton_data', picture_list[ins]))
            cropframe_skl = image_skl[p1[1]:p2[1],p1[0]:p2[0]]
            cv2.imwrite(os.path.join('/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/shiyandata/Skeleton_data_crop', picture_list[ins]), cropframe_skl)
        else:
            cv2.imwrite(os.path.join(output_dir,picture_list[ins]), image)
            image_skl = cv2.imread(os.path.join('/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/shiyandata/Skeleton_data', picture_list[ins]))
            cv2.imwrite(os.path.join('/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/shiyandata/Skeleton_data_crop', picture_list[ins]), image_skl)
            # print(picture_list[ins])
            # raise Exception()

        # else:
        #     if len(p_box) != 0:
        #         p_box = sorted(p_box, key=lambda e: e.__getitem__('weight_difference'), reverse=True)
        #         str_write = "{0} {1} {2} {3} {4} {5}\n".format(picture_list[ins],Category_labels_[picture_list[ins].split('_')[1]],max(p_box[0]['p1'][0],0), max(p_box[0]['p1'][1],0),min(p_box[0]['p2'][0],image.shape[1]), min(p_box[0]['p2'][1],image.shape[0]))
        #         f.write(str_write)
        #     else:
        #         os.remove(os.path.join(input_dir,picture_list[ins]))

##############################################################################
# 检测捕获的帧级图像
def eval_video(net, frame):
    global frame_list
    img = cv2.resize(frame, (300, 300))  # resize 300 * 300
    img = img - 127.5
    img = img * 0.007843  # 取均值
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img  # 将图片载入到blob中

    out = net.forward()
    box, conf, cls = postprocess(frame, out)
    return box, conf, cls

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
# 主函数
if __name__ =='__main__':
    build_net()     # 网络初始化
    picture_list = load_picture()     # 加载图像
    eval_picture(net_detection_)
