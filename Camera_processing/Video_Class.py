#coding=utf-8
# Video_Class.py           # 实时捕获摄像头输入

import cv2
import time
import sys
sys.path.append(r'/media/pro/PRO/Ubuntu/常用代码/网络测试/')

class Video:
    def CaptureVideo(self):
        cap = cv2.VideoCapture(0)
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            start = time.time()
            # Display the resulting frame
            cv2.imshow('Webcam', frame)
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

if __name__ =='__main__':
    V = Video()
    V.CaptureVideo()