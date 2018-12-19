#coding=utf-8
# Speech_Class.py           # 语音播报

import os
import threading

Speech_dic = {  0:'../Speech_processing/say_chest_pain1.wav',
                1:'../Speech_processing/say_chest_pain2.wav',
                2: '../Speech_processing/say_headache1.wav',
                3: '../Speech_processing/say_headache2.wav',
                4: '../Speech_processing/say_readphone.wav',
                5: '../Speech_processing/say_stomachache.wav',
                6: '../Speech_processing/say_toothache.wav',
              }

##############################################################################
# 自定义线程类
class myThread(threading.Thread):   # 继承父类threading.Thread
    def __init__(self, name,Speech_num):
        threading.Thread.__init__(self)
        self.name = name
        self.Speech_num = Speech_num
    def run(self):                   # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        print "Starting " + self.name
        P = Speech()
        P.PlaySpeech(self.Speech_num)
        print "Exiting " + self.name

# 启用自定义线程函数
def Thread_for_speech(Speech_num):
    # 创建新线程
    thread = myThread("Speech:%s" % Speech_dic[Speech_num][2:-4], Speech_num)
    # 开启线程
    thread.start()

##############################################################################
# 语音播报类
class Speech:
    def PlaySpeech(self,Speech_num):
        command = ['play','-q',Speech_dic[Speech_num]]
        command = ' '.join(command)
        os.system(command)
        return

# if __name__ =='__main__':
#     Thread_for_speech(0)
#     Thread_for_speech(1)
#     Thread_for_speech(2)
#     Thread_for_speech(3)
#     Thread_for_speech(4)
#     Thread_for_speech(5)
#     Thread_for_speech(6)