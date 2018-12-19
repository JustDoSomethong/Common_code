#-*- coding: UTF-8 -*-

import cv2
import os
import sys
import msvcrt

reload(sys)
sys.setdefaultencoding('u8')

# 当鼠标按下时变为True
drawing=False
x_1, y_1, x_2, y_2= -1,-1,-1,-1

# mouse callback function
def draw(event, x, y, flags, param):
    global x_1, y_1, x_2, y_2, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_1, y_1 = x, y
        # print x_1, y_1

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            x_2, y_2 = x, y
            # print x_2, y_2

def FindPicture(filepath):
    filepath = unicode(filepath, 'utf-8')
    pathDir = os.listdir(filepath)
    return pathDir

def Picture_filter(pathDir,start_Pic):
    Picture_filter_Dir = []
    for img_name in pathDir:
        Read_Flag = 0
        if img_name.endswith('_flip.jpg'):
            continue
        if img_name[0:4] != '2008':
            # print img_name[0:4]
            continue
        if int(img_name[5:len(img_name)-4]) < start_Pic:
            continue
        img_dir = os.path.join(filepath, img_name)
        xml_dir = os.path.join(xmlpath, img_name[0:len(img_name) - 4]) + '.xml'
        if os.path.exists(xml_dir):
            with open(xml_dir, 'r') as f:
                for line in f.readlines():
                    if line.find('<name>person</name>') != -1:
                        Read_Flag = 1
                        f.close()
                        break
        if Read_Flag == 0:
            continue
        Picture_filter_Dir.append(img_name)
    return Picture_filter_Dir

def Select_Pic(Picture_filter_Dir,name_num):
    i = 0
    while i < len(Picture_filter_Dir):
        img_name = Picture_filter_Dir[i]
        img_dir = os.path.join(filepath, img_name)
        img = cv2.imread(img_dir.decode('u8').encode('gbk'))

        cv2.namedWindow(img_name)
        cv2.setMouseCallback(img_name, draw)
        while True:
            cv2.imshow(img_name, img)
            key = cv2.waitKey(1) & 0xFF
            name_num_s = "%06d" % name_num
            res_name = name + name_num_s + '.jpg'
            # print res_name
            if key == ord(' '):
                name_num = name_num + 1
                if x_1 == -1 | y_1 == -1 | x_2 == -1 | y_2 == -1:
                    continue
                # print img.shape,x_1, y_1, x_2, y_2
                cropImg = img[y_1:y_2, x_1:x_2]
                while True:
                    cv2.imshow('image', cropImg)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('.'):
                        if name_num != 0:
                            name_num = name_num - 1
                        cv2.destroyAllWindows()
                        break
                    if key == ord('1'):
                        path_1 = os.path.join(respath, r'1_捂头', res_name)
                        cv2.imwrite(path_1.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('2'):
                        path_2 = os.path.join(respath, r'2_挠头', res_name)
                        cv2.imwrite(path_2.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('3'):
                        path_3 = os.path.join(respath, r'3_擦眼', res_name)
                        cv2.imwrite(path_3.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('4'):
                        path_4 = os.path.join(respath, r'4_捂脸', res_name)
                        cv2.imwrite(path_4.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('5'):
                        path_5 = os.path.join(respath, r'5_捂口', res_name)
                        cv2.imwrite(path_5.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('6'):
                        path_6 = os.path.join(respath, r'6_捂鼻', res_name)
                        cv2.imwrite(path_6.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('7'):
                        path_7 = os.path.join(respath, r'7_托腮', res_name)
                        cv2.imwrite(path_7.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('8'):
                        path_8 = os.path.join(respath, r'8_其他', res_name)
                        cv2.imwrite(path_8.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('s'):
                        path_s = os.path.join(respath, r's_正常站', res_name)
                        cv2.imwrite(path_s.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('d'):
                        path_d = os.path.join(respath, r'd_正常坐', res_name)
                        cv2.imwrite(path_d.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('r'):
                        path_r = os.path.join(respath, r'r_reading', res_name)
                        cv2.imwrite(path_r.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('b'):
                        path_b = os.path.join(respath, r'b_抱胸', res_name)
                        cv2.imwrite(path_b.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('f'):
                        path_f = os.path.join(respath, r'f_非抱胸', res_name)
                        cv2.imwrite(path_f.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('c'):
                        path_c = os.path.join(respath, r'c_cleaning', res_name)
                        cv2.imwrite(path_c.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                    if key == ord('p'):
                        path_p = os.path.join(respath, r'p_phone', res_name)
                        cv2.imwrite(path_p.decode('u8').encode('gbk'), cropImg)
                        cv2.destroyAllWindows()
                        break
                cv2.destroyAllWindows()
                break
            if key == ord('.'):
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord(','):
                cv2.destroyAllWindows()
                i = i - 1
                break
            if key == ord('1'):
                name_num = name_num + 1
                path_1 = os.path.join(respath, r'1_捂头', res_name)
                cv2.imwrite(path_1.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('2'):
                name_num = name_num + 1
                path_2 = os.path.join(respath,r'2_挠头', res_name)
                cv2.imwrite(path_2.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('3'):
                name_num = name_num + 1
                path_3 = os.path.join(respath,r'3_擦眼', res_name)
                cv2.imwrite(path_3.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('4'):
                name_num = name_num + 1
                path_4 = os.path.join(respath,r'4_捂脸', res_name)
                cv2.imwrite(path_4.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('5'):
                name_num = name_num + 1
                path_5 = os.path.join(respath,r'5_捂口', res_name)
                cv2.imwrite(path_5.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('6'):
                name_num = name_num + 1
                path_6 = os.path.join(respath,r'6_捂鼻', res_name)
                cv2.imwrite(path_6.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('7'):
                name_num = name_num + 1
                path_7 = os.path.join(respath,r'7_托腮', res_name)
                cv2.imwrite(path_7.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('8'):
                name_num = name_num + 1
                path_8 = os.path.join(respath,r'8_其他', res_name)
                cv2.imwrite(path_8.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('s'):
                name_num = name_num + 1
                path_s = os.path.join(respath,r's_正常站', res_name)
                cv2.imwrite(path_s.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('d'):
                name_num = name_num + 1
                path_d = os.path.join(respath,r'd_正常坐', res_name)
                cv2.imwrite(path_d.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('r'):
                name_num = name_num + 1
                path_r = os.path.join(respath,r'r_reading', res_name)
                cv2.imwrite(path_r.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('b'):
                name_num = name_num + 1
                path_b = os.path.join(respath,r'b_抱胸', res_name)
                cv2.imwrite(path_b.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('f'):
                name_num = name_num + 1
                path_f = os.path.join(respath,r'f_非抱胸', res_name)
                cv2.imwrite(path_f.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('c'):
                name_num = name_num + 1
                path_c = os.path.join(respath,r'c_cleaning', res_name)
                cv2.imwrite(path_c.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break
            if key == ord('p'):
                name_num = name_num + 1
                path_p = os.path.join(respath,r'p_phone', res_name)
                cv2.imwrite(path_p.decode('u8').encode('gbk'), img)
                cv2.destroyAllWindows()
                i = i + 1
                break

if __name__ == '__main__':
    filepath =r'I:\Ubuntu\重要资料\caffe数据库和模型\数据库\VOC\2012\VOC2012\JPEGImages'
    xmlpath =r'I:\Ubuntu\重要资料\caffe数据库和模型\数据库\VOC\2012\VOC2012\Annotations'
    respath =r'C:\Users\39294\Desktop\PPT\VOC'
    name='VOC2008_'
    start_Pic = 0
    name_num = 0
    Read_Flag = 0
    pathDir =  FindPicture(filepath)
    Picture_filter_Dir = Picture_filter(pathDir,start_Pic)
    Select_Pic(Picture_filter_Dir,name_num)