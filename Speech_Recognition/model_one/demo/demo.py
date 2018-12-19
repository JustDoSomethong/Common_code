#coding=utf-8
import os
import cv2
import sys
import threading
sys.path.append('./YH/')
from CLASS_YH import *
sys.path.append('./yhq/')
from neaten import *
sys.path.append('./yk/')
from yk_models import *

##############################################################################
class myThread(threading.Thread):   # 继承父类threading.Thread
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        print "Starting " + self.name
        os.system('./asr_record_sample')
        print "Exiting " + self.name

def Thread_for_SDK():
    # 创建新线程
    thread = myThread("SDk")
    # 开启线程
    thread.start()
##############################################################################

if __name__ =='__main__':
    Thread_for_SDK()
    cap = cv2.VideoCapture(0)
    YH = YH_detection_classification()
    facedetec, landmark = gazenet_init()
    yk = yk_models()
    prev_mode = 0
    mode = 0
    while (1):
        ret, frame = cap.read()
        for line in open("./SDK_result.txt",'r'):
            if line[7:14] == "contact":
                if line[19:24] == "65336":
                    mode = 1
                    # print line[19:24]
                if line[19:24] == "65337":
                    mode = 2
                    # print line[19:24]
                if line[19:24] == "65338":
                    mode = 3
                    # print line[19:24]
        if mode != prev_mode:
            cv2.destroyAllWindows()
            if mode != 2:
                yk.refresh()
        prev_mode = mode
        if mode == 1:
            gazedetect(frame, facedetec, landmark)
        elif mode == 2:
            yk.main(frame)
        elif mode ==3:
            YH.forward(frame)
        if cv2.waitKey(1) & 0xFF == ord('z'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # exit()



