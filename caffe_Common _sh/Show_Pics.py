#-*- coding: UTF-8 -*-
import os
import cv2
import numpy as np

filepath ='/media/pro/PRO/Ubuntu/重要资料/caffe数据库和模型/数据库/COCO/coco2017/images/val2017'

pathDir = os.listdir(filepath)
# print pathDir[1]

# for i in range(len(pathDir)):
#     img_dir = os.path.join(filepath, pathDir[i])
#     # print img_dir
#     img = cv2.imread(img_dir)
#     while True:
#         cv2.imshow('1', img)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord(' '):
#             cv2.destroyAllWindows()
#             break

img_1 = cv2.imread(os.path.join(filepath, pathDir[1]))
img_2 = cv2.imread(os.path.join(filepath, pathDir[2]))

print img_1.shape[0],img_1.shape[1],img_2.shape[0],img_2.shape[1]
merge_img = np.zeros((max(img_1.shape[0],img_2.shape[0]),img_1.shape[1]+img_2.shape[1],3), np.uint8)
merge_img[...] = 0
merge_img[0:img_1.shape[0],0:img_1.shape[1]] = img_1
merge_img[0:img_2.shape[0]:,img_1.shape[1]:] = img_2

while True:
    cv2.imshow('1', merge_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        cv2.destroyAllWindows()
        break