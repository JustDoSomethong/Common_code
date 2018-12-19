# -*- coding: utf-8 -*-
import sys
sys.path.append("/home/pro/caffe-master/python")
sys.path.append("/home/pro/caffe-master/python/caffe")
import caffe
from caffe import layers as L,params as P,proto,to_proto
#设定文件的保存路径
test_proto='/home/pro/Project/NanRi/inception_elec_open_close/deploy_inception_v4.prototxt'     #训练配置文件  
cmodel='/home/pro/Project/NanRi/inception_elec_open_close/_iter_50000.caffemodel'
test_proto_classify='/home/pro/Project/NanRi/inception_elec_open_close/deploy_inception_v4_classify.prototxt'
cmodel_classsif='/home/pro/Project/NanRi/inception_elec_open_close/_iter_20000.caffemodel'
#开始训练
def testing():
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.Net(test_proto,cmodel,caffe.TEST)
    
    for i in range(10):
        solver.forward()
        ests = solver.blobs['classifiernew'].data > 0
        print ests

def testing2():
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver_classify = caffe.Net(test_proto_classify,cmodel_classsif,caffe.TEST)
    for i in range(10):
        solver_classify.forward()
        ests_classify = solver_classify.blobs['class'].data > 0.5
        print ests_classify
if __name__ == '__main__':
    testing()
    testing2()
