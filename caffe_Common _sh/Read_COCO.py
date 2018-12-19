#-*- coding: UTF-8 -*-
import json
import os
import cv2

# 本段代码读取json文件,选取COCO数据集中为person的类,并显示出来
filepath ='/media/pro/PRO/Ubuntu/重要资料/caffe数据库和模型/数据库/COCO/coco2017/images/val2017'
with open("/media/pro/PRO/Ubuntu/重要资料/caffe数据库和模型/数据库/COCO/coco2017/annotations/instances_val2017.json",'r') as f:
    load_json = json.load(f)

# 显示文件夹下图像
# pathDir = os.listdir(filepath)
# img_dir = os.path.join(filepath,pathDir[1])
# img = cv2.imread(img_dir)
# cv2.imshow('1', img)

# 根据json文件,读取类别为person(category_id=1)的图像
image = load_json['images']
img_annotations = load_json['annotations']
for i in range(len(img_annotations)):
    if img_annotations[i]['category_id'] != 1:
        continue
    img_id = img_annotations[i]['image_id']
    img_name = str(img_id).zfill(12)
    img_dir = os.path.join(filepath,img_name)+'.jpg'
    img = cv2.imread(img_dir)
    cv2.namedWindow(img_name)
    while True:
        cv2.imshow(img_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('.'):
            cv2.destroyAllWindows()
            break
