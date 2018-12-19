import sys, os, cv2, re
import numpy as np

caffe_root = '/home/pro/caffe-ssd'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe


class TraceClassifier():
    def __init__(self,
                 DeployFile='/home/yinkang/0MyProject/0_HandSSD/TraceClassification/mobilenet_v1/mobilenet_deploy_depth.prototxt',
                 CaffeModel='/home/yinkang/0MyProject/0_HandSSD/TraceClassification/mobilenet_v1/snapshot/mobilnet_iter_68000.caffemodel',
                 LabelMap='/home/yinkang/0MyProject/0_HandSSD/TraceClassification/label_map.txt',
                 MeanFile=None,
                 UseGPU=True):
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
        img = self.Transformer.preprocess('data', img)
        img /= 255.0
        self.Net.blobs['data'].data[...] = img

        self.Net.forward()
        self.pred = self.Net.blobs['prob'].data.argmax()
        self.conf = self.Net.blobs['prob'].data.max()


if __name__ == '__main__':
    img = cv2.imread('/home/yinkang/0MyProject/0_HandSSD/trace.png')

    classifier = TraceClassifier()
    classifier.forward(img)

    cv2.putText(img, '%s' % classifier.label_map[classifier.pred], (40, 60), cv2.FONT_ITALIC, 2, (0, 0, 255), 1)
    cv2.imshow('img', img)
    cv2.waitKey()
