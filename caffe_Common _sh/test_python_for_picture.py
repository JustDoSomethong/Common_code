# coding=utf-8

caffe_root = '/home/kb539/caffe'
import sys

sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
import os

deploy = '/home/kb539/YH/inception_elec_open_close/deploy_inception_v4.prototxt'
deploy_classify = '/home/kb539/YH/inception_elec_open_close/deploy_inception_v4_classify.prototxt'
caffe_model = '/home/kb539/YH/inception_elec_open_close/_iter_50000.caffemodel'
caffe_model_classify = '/home/kb539/YH/inception_elec_open_close/_iter_20000.caffemodel'
img_path = '/home/kb539/YH/inception_elec_open_close/picture/2号主变220千伏正母闸刀静触头B相2016-08-20060332.jpg'

meanfile_path = '/home/kb539/YH/inception_elec_open_close/imagenet_mean.binaryproto' #可选

txtName = "test.txt"

net = caffe.Net(deploy, caffe_model, caffe.TEST)

# 图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 设定维度格式(Num, Channels, Height, Width)
transformer.set_transpose('data', (2, 0, 1))  # 改变维度的顺序 (height,width,channels)->(channels,height,width)

if os.path.exists(meanfile_path):  # 减去均值
    FORMAT = os.path.splitext(meanfile_path)[-1]
    if FORMAT == '.npy':
        transformer.set_mean('data', np.load(meanfile_path).mean(1).mean(1))
    elif FORMAT == '.binaryproto':
        blob = caffe.proto.caffe_pb2.BlobProto()
        mean = open(meanfile_path, 'rb').read()
        blob.ParseFromString(mean)
        data = np.array(caffe.io.blobproto_to_array(blob))
        mean_npy = data[0]
        transformer.set_mean('data', mean_npy.mean(1).mean(1))

transformer.set_raw_scale('data', 255)  # 缩放到[0，255]
transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，RGB->BGR

im = caffe.io.load_image(img_path)  # 加载图片

net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行预处理操作，并将图片载入到blob中

# 执行测试
out = net.forward()

prob = out['classifiernew'][-1] > 0

print '\nprob : ', prob
f = file(txtName,"a+")
f.write("test_12:\t")
for i in range(12):
    if prob[i] > 0:
        f.write("1\t")
    else:
        f.write("0\t")
if prob[7] > 0:
    f.write("\ntest_sub_classify:\t")
    net_classify = caffe.Net(deploy_classify, caffe_model_classify, caffe.TEST)
    net_classify.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行预处理操作,载入图片
    out_classify = net_classify.forward()
    sub_classify = out_classify['class'][-1] > 0.5
    print sub_classify
    for i in range(6):
        if sub_classify[i] > 0:
            f.write("1\t")
        else:
            f.write("0\t")
f.write("\n")
f.close()
