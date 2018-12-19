# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import dlib
import sys, os, time

caffe_root = '/home/pro/caffe-ssd'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe


def FPS(func):
    def wrapper(*args, **kwargs):
        time_start = time.time()
        func(*args, **kwargs)
        time_end = time.time()
        fps = 1.0 / (time_end - time_start)
        print('FPS = %.2f' % fps)

    return wrapper


def TIME(func):
    def wrapper(*args, **kwargs):
        time_start = time.time()
        func(*args, **kwargs)
        time_end = time.time()
        fps = 1.0 / (time_end - time_start)
        print('running time of %s = %.2f ms' % (func.__name__, (time_end - time_start) * 1000))

    return wrapper


class Detecter():

    def __init__(self, DeployFile='/home/yinkang/0MyProject/0_HandSSD/MobileNetSSD_deploy3.prototxt',
                 CaffeModel='/home/yinkang/0MyProject/0_HandSSD/mobilenet_iter_10000.caffemodel',
                 Class=('background', 'hand', 'fist'), UseGPU=True):
        self.net = caffe.Net(DeployFile, CaffeModel, caffe.TEST)
        if UseGPU:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.iou__threshold = 0.5
        self.conf_threshold = 0.5

        self.img = None
        self.box = None
        self.conf = None
        self.cls = None
        self.deleted = None
        self.orig_shape = None
        self.classes = Class
        self.class_max = len(self.classes) - 1
        self.detecting_none_target_frames = 0

        self.tracker = dlib.correlation_tracker()
        self.tracking_result = None
        self.tracking_mode = False
        self.tracking_confidence = 0.0
        self.tracking_threshold = 6
        self.tracking_box = None
        self.tracking_drift_frames = 0

    def detect(self, origimg):
        if not self.tracking_mode:
            self.tracking_drift_frames += 1
            self.preprocess(origimg)
            self.net.blobs['data'].data[...] = self.img
            out = self.net.forward()['detection_out']
            self.postprocess(out)
            self.NMS()
            if len(self.conf) > 0 and self.conf[0] > 0.90:
                self.set_track_traget(origimg, 0)
        else:
            self.tracking_drift_frames = 0
            self.track(origimg)

    def detect_v2(self, origimg):
        self.preprocess(origimg)
        self.net.blobs['data'].data[...] = self.img
        out = self.net.forward()['detection_out']
        self.postprocess(out)
        self.NMS()
        if not len(self.conf):
            self.detecting_none_target_frames += 1
        else:
            self.detecting_none_target_frames = 0

        if self.detecting_none_target_frames > 30:
            self.tracking_mode = False
        if not self.tracking_mode:
            self.tracking_drift_frames += 1
            if len(self.conf) and self.conf[0] > 0.9:
                self.set_track_traget(origimg, 0)
        else:
            self.tracking_drift_frames = 0
            self.track(origimg)

    def detect_only(self, origimg):
        self.preprocess(origimg)
        self.net.blobs['data'].data[...] = self.img
        out = self.net.forward()['detection_out']
        self.postprocess(out)
        self.NMS()

    ##################################################################
    ################ pre-post process
    def preprocess(self, origimg):
        self.img = cv2.resize(origimg, (300, 300))
        self.img = self.img - 127.5
        self.img = self.img * 0.007843
        self.img = self.img.astype(np.float32)
        self.img = self.img.transpose((2, 0, 1))
        self.orig_shape = origimg.shape

    def postprocess(self, out):
        h = self.orig_shape[0]
        w = self.orig_shape[1]

        self.box = out[0, 0, :, 3:7] * np.array([w, h, w, h])
        self.cls = out[0, 0, :, 1]
        self.conf = out[0, 0, :, 2]
        for i in range(len(self.conf)):
            if self.conf[i] < self.conf_threshold:
                self.box = self.box[:i, :]
                self.conf = self.conf[:i]
                self.cls = self.cls[:i]
                break
        self.box = self.box.astype(np.int32)

    ##################################################################
    ############    NMS
    def IOU(self, bbox1, bbox2):
        contain = bbox1 - bbox2
        if (contain[0] >= 0 and contain[1] >= 0 and contain[2] <= 0 and contain[3] <= 0) \
                or (contain[0] <= 0 and contain[1] <= 0 and contain[2] >= 0 and contain[3] >= 0):
            return True
        if bbox1[2] > bbox2[0]:
            w = bbox1[2] - bbox2[0]
            h = bbox1[3] - bbox2[1]
        else:
            w = bbox2[2] - bbox1[0]
            h = bbox2[3] - bbox1[1]
        if w <= 0 or h <= 0:
            return False
        s1 = (bbox1[2] - bbox1[2]) * (bbox1[3] - bbox1[1])
        s2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        i = float(w * h)
        iou = i / (s1 + s2 - i)
        return iou > self.iou__threshold

    # two classes: hand , fist
    def NMS(self):
        n = len(self.cls)
        self.deleted = [False for i in range(n)]
        if n > 1:
            loc = [list() for i in range(self.class_max)]
            for i in range(n):
                loc[int(self.cls[i]) - 1].append(i)
            for l in loc:
                n = len(l)
                if n > 1:
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (not self.deleted[l[j]]) and self.IOU(self.box[l[i]], self.box[l[j]]):
                                self.deleted[l[j]] = True

    ##################################################################
    ################### tracking

    def set_track_traget(self, img, index_of_bbox):
        self.tracking_mode = True
        self.tracking_box = [int(self.box[index_of_bbox, 0]), int(self.box[index_of_bbox, 1]),
                             int(self.box[index_of_bbox, 2]), int(self.box[index_of_bbox, 3])]
        self.tracker.start_track(img, dlib.rectangle(self.tracking_box[0], self.tracking_box[1],
                                                     self.tracking_box[2], self.tracking_box[3]))
    @TIME
    def track(self, img):
        self.tracking_confidence = self.tracker.update(img)
        self.tracking_mode = self.tracking_confidence >= self.tracking_threshold
        if self.tracking_mode:
            bbox = self.tracker.get_position()
            self.tracking_box = [int(bbox.left()), int(bbox.top()), int(bbox.right()), int(bbox.bottom())]
            if len(self.conf):
                for i in range(len(self.conf)):
                    if not self.deleted[i]:
                        if self.cmp_tracking_detecting_box(i):
                            self.set_track_traget(img, i)
                            break

    def cmp_tracking_detecting_box(self, index_of_detecing_box):
        tracking_box = np.array(self.tracking_box)
        detecting_box = self.box[index_of_detecing_box, :]
        return self.IOU(tracking_box, detecting_box)
