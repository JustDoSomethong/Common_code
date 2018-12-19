#-*- coding: UTF-8 -*-
# rename.py           # 对文件夹下子文件中图像进行重命名,并生成全新的文件,格式为img_类别_id.jpg

import os
import cv2

##################################################################
# 对文件夹下子文件中图像进行重命名，读取图像，随后保存至不同文件夹
def Rename_for_Category_labels_(input_dir,output_dir,image_format='jpg',trim_format='%06d'):
    dataset = os.listdir(input_dir)
    for in_dir in dataset:
        res_dir = []
        for filename in os.listdir(os.path.join(input_dir, in_dir)):
            if filename.endswith(image_format):
                img = cv2.imread(os.path.join(input_dir, in_dir, filename))
                res_dir.append(img)
        print (in_dir,',len = %3d'%len(res_dir))
        for i in range(len(res_dir)):
            res_name = str('img_' + in_dir + "_%06d" % i + '.jpg')
            cv2.imwrite(os.path.join(output_dir,in_dir, res_name), res_dir[i])
    print ("Rename finished!!")
    return
##################################################################
# 对文件夹下子文件中图像进行重命名，在同一个文件夹下操作
def Rename_for_Category_labels(input_dir,image_format='png',trim_format='%06d'):
    dataset = os.listdir(input_dir)
    for in_dir in dataset:
        for filename in os.listdir(os.path.join(input_dir, in_dir)):
            if filename.endswith(image_format):
                os.rename(os.path.join(input_dir, in_dir,filename),os.path.join(input_dir, in_dir,filename[0:-13]+'.png'))
        print ("%s's files has been done!!" % in_dir)
    print ("Rename finished!!")
    return
##################################################################
# 对文件夹中图像进行重命名，在同一个文件夹下操作
def Rename_for_folders(input_dir,image_format='jpg',trim_format='%06d'):
    for filename in os.listdir(input_dir):
        if filename.endswith(image_format):
             os.rename(os.path.join(input_dir, filename),os.path.join(input_dir, filename + '_?.jpg'))
    print ("Rename finished!!")
    return

##################################################################
# 对文件夹中图像读取图像，随后保存至不同文件夹
def Change_folders(input_dir,output_dir,image_format='jpg'):
    for filename in os.listdir(input_dir):
        if filename.endswith(image_format):
            img = cv2.imread(os.path.join(input_dir,filename))
            cv2.imwrite(os.path.join(output_dir, filename), img)
    print ("Rename finished!!")
    return

if __name__ == '__main__':
    Change_folders('./detection_data/010','./JPEGImages',)
