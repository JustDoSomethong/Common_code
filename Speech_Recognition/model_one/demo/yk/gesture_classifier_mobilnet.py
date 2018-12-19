import sys, os, cv2, re
import numpy as np
import matplotlib.pyplot as plt

caffe_root = '/home/pro/caffe-ssd'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe


class GestureClassifier():
    def __init__(self, DeployFile, CaffeModel, LabelMap, MeanFile=None, UseGPU=True):
        self.pred = 0
        self.conf = 0
        self.Net = caffe.Net(DeployFile, CaffeModel, caffe.TEST)
        self.label_map = dict()
        for line in open(LabelMap, 'r'):
            line = line.strip()
            line = re.split(r'\s+', line)
            if len(line) == 2:
                self.label_map[int(line[1])] = line[0]

        self.Transformer = caffe.io.Transformer({'data': self.Net.blobs['data'].data.shape})
        self.Transformer.set_transpose('data', (2, 0, 1))
        if MeanFile is not None:
            self.Transformer.set_mean('data', np.load(MeanFile).mean(1).mean(1))
        # self.Transformer.set_raw_scale('data', 0.00390625)
        self.Transformer.set_channel_swap('data', (2, 1, 0))
        if UseGPU:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

    def forward(self, img):
        # img = img[:, 50:-50, :]
        # img = cv2.resize(img, (224, 224))
        # img = img.astype(np.float32)
        # img = img.transpose((2, 0, 1))
        # img = img[::-1, :, :]
        # img *= 0.00390625

        # shape = img.shape
        # h, w = shape[0], shape[1]
        # if h > w:
        #     delta = (h - w) / 2
        #     blank = np.zeros(shape=(h, delta, 3), dtype=np.uint8) + 255
        #     img = np.hstack([blank, img, blank])
        # elif h < w:
        #     delta = (w - h) / 2
        #     blank = np.zeros(shape=(delta, w, 3), dtype=np.uint8) + 255
        #     img = np.vstack([blank, img, blank])
        shape = img.shape
        h, w = shape[0], shape[1]
        if h > w:
            delta = (h - w) / 2
            blank = np.zeros(shape=(h, delta, 3), dtype=np.uint8) + 255
            img = np.hstack([blank, img, blank])
        elif h < w:
            delta = (w - h) / 2
            blank = np.zeros(shape=(delta, w, 3), dtype=np.uint8) + 255
            img = np.vstack([blank, img, blank])

        img = self.Transformer.preprocess('data', img)
        img /= 255.0
        self.Net.blobs['data'].data[...] = img

        self.Net.forward()
        self.pred = self.Net.blobs['prob'].data.argmax()
        self.conf = self.Net.blobs['prob'].data.max()


if __name__ == '__main__':

    # net = caffe.Net('/home/yinkang/0MyProject/0_HandSSD/TraceClassification/finger_loc/test/mobilenet_trainval_depth.prototxt', '/home/yinkang/0MyProject/0_HandSSD/TraceClassification/finger_loc/snapshot/gesture_iter_12000.caffemodel', caffe.TEST)
    # net.forward()
    # im = net.blobs['data'].data
    # im = np.squeeze(im)
    # im = np.transpose(im, (1, 2, 0))
    # plt.imshow(im)
    # plt.show()
    #
    # exit()




    img = cv2.imread('/home/yinkang/0MyProject/0_HandSSD/gesture.png')

    classifier = GestureClassifier(
        DeployFile='/home/yinkang/0MyProject/0_HandSSD/TraceClassification/finger_loc/finger_classification.prototxt',
        CaffeModel='/home/yinkang/0MyProject/0_HandSSD/TraceClassification/finger_loc/snapshot/gesture_iter_12000.caffemodel',
        LabelMap='/home/yinkang/0MyProject/0_HandSSD/TraceClassification/finger_loc/gesture_label_map.txt',
        MeanFile='/home/yinkang/0MyProject/0_HandSSD/TraceClassification/finger_loc/gesture_mean_blank.npy',
        UseGPU=True)
    classifier.forward(img)

    # cv2.putText(img, '%s' % classifier.label_map[classifier.pred], (40, 60), cv2.FONT_ITALIC, 2, (0, 0, 255), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    im = classifier.Net.blobs['data'].data
    im = np.squeeze(im)
    im = np.transpose(im, (1, 2, 0))
    print type(im)
    plt.imshow(im)
    plt.show()
