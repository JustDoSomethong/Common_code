#-*- coding: UTF-8 -*-
# Resize_Crop.py           # 对文件夹下子文件中图像进行缩放并crop,保证缩放后图像没有发生拉伸

import os
import cv2

##################################################################
# 对文件夹下子文件中图像进行缩放并crop
def Resize_Crop_for_Images(input_dir,image_crop_size = 224,image_format='jpg'):
    image_crop_size = float(image_crop_size)
    dataset = os.listdir(input_dir)
    for in_dir in dataset:
        for filename in os.listdir(os.path.join(input_dir, in_dir)):
            if filename.endswith(image_format):
                img = cv2.imread(os.path.join(input_dir, in_dir, filename))
                height_and_weight = img.shape
                # print filename
                # print height_and_weight[0],height_and_weight[1]
                if height_and_weight[0] > height_and_weight[1]:
                    if height_and_weight[1] >= image_crop_size:
                        height = height_and_weight[0]/2
                        weight = height_and_weight[1]/2
                        crop = img[height - int(image_crop_size / 2):height + (int(image_crop_size / 2) - 1),
                               weight - int(image_crop_size / 2):weight + (int(image_crop_size / 2) - 1)]
                    else:
                        resize_ratio = float(float(image_crop_size) / float(height_and_weight[1]))
                        img = cv2.resize(img,(0,0),fx=resize_ratio,fy=resize_ratio,interpolation=cv2.INTER_CUBIC)
                        height_and_weight = img.shape
                        height = height_and_weight[0] / 2
                        weight = height_and_weight[1] / 2
                        crop = img[height - int(image_crop_size / 2):height + (int(image_crop_size / 2) - 1),
                               weight - int(image_crop_size / 2):weight + (int(image_crop_size / 2) - 1)]
                else:
                    if height_and_weight[0] >= image_crop_size:
                        height = height_and_weight[0]/2
                        weight = height_and_weight[1]/2
                        crop = img[height - int(image_crop_size / 2):height + (int(image_crop_size / 2) - 1),
                               weight - int(image_crop_size / 2):weight + (int(image_crop_size / 2) - 1)]
                    else:
                        resize_ratio = float(float(image_crop_size) / float(height_and_weight[0]))
                        img = cv2.resize(img,(0,0),fx=resize_ratio,fy=resize_ratio,interpolation=cv2.INTER_CUBIC)
                        height_and_weight = img.shape
                        height = height_and_weight[0] / 2
                        weight = height_and_weight[1] / 2
                        crop = img[height - int(image_crop_size/2):height + (int(image_crop_size/2)-1),
                               weight - int(image_crop_size/2):weight + (int(image_crop_size/2)-1)]
                cv2.imwrite(os.path.join(input_dir,in_dir,filename), crop)
    print ("Resize and Crop finished!!")
    return
##################################################################
