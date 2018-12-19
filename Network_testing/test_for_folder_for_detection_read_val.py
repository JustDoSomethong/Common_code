import numpy as np
import sys, os
import cv2

caffe_root = '/home/pro/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')
import caffe

net_file = '/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/SSD_data/snapshot/deploy.prototxt'
caffe_model = '/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/SSD_data/snapshot/mobilenet_iter_40000.caffemodel'
val_txt = '/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/SSD_data/ImageSets/Main/val.txt'
img_dir = '/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/SSD_data/JPEGImages'
result_dir = '/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/SSD_data/dection_result'

net = caffe.Net(net_file, caffe_model, caffe.TEST)

CLASSES = ('background',
           'eating', 'drinking', 'calling', 'reading',
           'headache', 'toothache', 'chest_pain', 'waist_pain', 'stomachache',
           'falling', 'others')


def preprocess(src):
    img = cv2.resize(src, (300, 300))
    img = img - 127.5
    img = img * 0.007843
    return img


def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0, 0, :, 1]
    conf = out['detection_out'][0, 0, :, 2]
    return (box.astype(np.int32), conf, cls)


def detect(imgfile):
    print os.path.join(result_dir,imgfile)
    origimg = cv2.imread(os.path.join(img_dir,imgfile))
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(origimg, p1, p2, (0, 255, 0))
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    # cv2.imshow("SSD", origimg)
    cv2.imwrite(os.path.join(result_dir,imgfile),origimg)
    # k = cv2.waitKey(0) & 0xff
    # Exit if ESC pressed
    # if k == 27: return False
    # if k == '.': return True
    return True


if __name__ in '__main__':
    file = open(val_txt,'r')
    lines = file.readlines()
    for line in lines:
        detect(line[:-1]+'.jpg')
