#-*- coding: UTF-8 -*-
# Data_Augmentation.py  # 数据增强模块,如:颜色抖动增加模型对光照强度的变化

import numpy as np
import os
from PIL import Image,ImageEnhance

##################################################################
# 对图像进行颜色抖动,增加对不同光照条件的鲁棒性
def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

def Data_Augmentation_for_folders(input_dir,image_format='jpg',category_num = 5):
    for filename in os.listdir(input_dir):
        if filename.endswith(image_format):
            for i in range(1,category_num):
                img = Image.open(os.path.join(input_dir, filename), mode="r")
                filename_res = randomColor(img)
                filename_res.save(os.path.join(input_dir, filename[0:-4]) + '_c%03d.jpg' % i)

def Data_Augmentation_(input_dir,image_format='jpg',category_num = 5):
    print ("Data_Augmentation_ begin!!")
    dataset = os.listdir(input_dir)
    for in_dir in dataset:
        Data_Augmentation_for_folders(os.path.join(input_dir, in_dir),image_format,category_num)
    print ("Data_Augmentation_ done!!")
