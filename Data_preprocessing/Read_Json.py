#-*- coding: UTF-8 -*-
# Read_Json.py		# 根据json文件，删除文件

import os
import json
import cv2
import numpy as np
import math

def json_list():
    list = os.listdir('./openpose_data/JSON')
    return list

def read_json(list):
    for img_json in list:
        file = open(os.path.join('./openpose_data/JSON',img_json),'r')
        # print(img_json)
        img_keypoints = json.load(file)
        if len(img_keypoints['people']) == 0:
            # print(img_json.split('_')[1],img_json.split('_')[2])
            img_dir = os.path.join('./openpose_data/Original_Data',img_json.split('_')[1],('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] +'.jpg'))
            if os.path.exists(img_dir):
                print(img_dir)
                os.remove(img_dir)
            img_keypoints_dir = os.path.join('./openpose_data/Original_Skeleton_Images', ('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
            if os.path.exists(img_keypoints_dir):
                print(img_keypoints_dir)
                os.remove(img_keypoints_dir)
            keypoints_dir = os.path.join('./openpose_data/Skeleton_Images', ('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
            if os.path.exists(keypoints_dir):
                print(keypoints_dir)
                os.remove(keypoints_dir)
            Skeleton_Images_drawn_dir = os.path.join('./openpose_data/Skeleton_Images_drawn', ('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
            if os.path.exists(Skeleton_Images_drawn_dir):
                print(Skeleton_Images_drawn_dir)
                os.remove(Skeleton_Images_drawn_dir)
        elif len(img_keypoints['people']) > 1:
            # print(img_json.split('_')[1],img_json.split('_')[2])
            img_dir = os.path.join('./openpose_data/Original_Data',img_json.split('_')[1],('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] +'.jpg'))
            if os.path.exists(img_dir):
                print(img_dir)
                os.remove(img_dir)
            img_keypoints_dir = os.path.join('./openpose_data/Original_Skeleton_Images', ('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
            if os.path.exists(img_keypoints_dir):
                print(img_keypoints_dir)
                os.remove(img_keypoints_dir)
            keypoints_dir = os.path.join('./openpose_data/Skeleton_Images', ('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
            if os.path.exists(keypoints_dir):
                print(keypoints_dir)
                os.remove(keypoints_dir)
            Skeleton_Images_drawn_dir = os.path.join('./openpose_data/Skeleton_Images_drawn', ('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
            if os.path.exists(Skeleton_Images_drawn_dir):
                print(Skeleton_Images_drawn_dir)
                os.remove(Skeleton_Images_drawn_dir)
        else:
            keypoints = img_keypoints['people'][0]['pose_keypoints']
            keypoints_nums = np.sum([x!=0 for x in keypoints])/3
            if keypoints_nums < 10:
                img_dir = os.path.join('./openpose_data/Original_Data', img_json.split('_')[1],
                                       ('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '.jpg'))
                if os.path.exists(img_dir):
                    print(img_dir)
                    os.remove(img_dir)
                img_keypoints_dir = os.path.join('./openpose_data/Original_Skeleton_Images', (
                'img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
                if os.path.exists(img_keypoints_dir):
                    print(img_keypoints_dir)
                    os.remove(img_keypoints_dir)
                keypoints_dir = os.path.join('./openpose_data/Skeleton_Images', (
                'img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
                if os.path.exists(keypoints_dir):
                    print(keypoints_dir)
                    os.remove(keypoints_dir)
                Skeleton_Images_drawn_dir = os.path.join('./openpose_data/Skeleton_Images_drawn', ('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
                if os.path.exists(Skeleton_Images_drawn_dir):
                    print(Skeleton_Images_drawn_dir)
                    os.remove(Skeleton_Images_drawn_dir)
            # else:
            #     write_point(img_json, img_keypoints)
        img_dir = os.path.join('./openpose_data/Original_Data', img_json.split('_')[1],('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '.jpg'))
        if os.path.exists(img_dir):
            img = cv2.imread(img_dir)
            if img.shape[0]<100 or img.shape[1]<100:
                if os.path.exists(img_dir):
                    print(img_dir)
                    os.remove(img_dir)
                img_keypoints_dir = os.path.join('./openpose_data/Original_Skeleton_Images', (
                'img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
                if os.path.exists(img_keypoints_dir):
                    print(img_keypoints_dir)
                    os.remove(img_keypoints_dir)
                keypoints_dir = os.path.join('./openpose_data/Skeleton_Images', (
                'img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
                if os.path.exists(keypoints_dir):
                    print(keypoints_dir)
                    os.remove(keypoints_dir)
                Skeleton_Images_drawn_dir = os.path.join('./openpose_data/Skeleton_Images_drawn', ('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
                if os.path.exists(Skeleton_Images_drawn_dir):
                    print(Skeleton_Images_drawn_dir)
                    os.remove(Skeleton_Images_drawn_dir)
    return

def write_point(img_json,img_keypoints):
    colors = [ ( 84,  0,255),(  0,  0,255),(  0, 84,255),(  0,167,255),(  0,255,255),(  0,255,167),(  0,255, 84),(  0,255,  0),( 84,255,  0),(167,255,  0),(255,255,  0),(255,167,  0),(255, 84,  0),(252,  0,  0),(167,  0,255),(255,  0,167),(255,  0,255),(255,  0, 84)]
    keypoints = img_keypoints['people'][0]['pose_keypoints']
    keypoints_list = [keypoints[i:i + 3] for i in range(0, len(keypoints), 3)]
    img_dir = os.path.join('./openpose_data/Skeleton_Images',('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg'))
    # print(img_dir)
    img = cv2.imread(img_dir)
    # print(img.shape)
    img_sk = np.zeros(img.shape,dtype=np.int8)
    write_connection(keypoints_list, img_sk)
    for i,[x,y,c] in enumerate(keypoints_list):
        # print(i,x,y,c)
        if c > 0:
            # print(img[int(y), int(x)])
            cv2.circle(img_sk, (int(x),int(y)), 3, colors[i], 6)
    cv2.imwrite(os.path.join('./openpose_data/Skeleton_Images_drawn',('img_' + img_json.split('_')[1] + '_' + img_json.split('_')[2] + '_rendered.jpg')),img_sk)
    # cv2.imshow('img',img_sk)
    # cv2.wait///////////Key()

def write_connection(keypoints_list,img_sk):
    if keypoints_list[15][2] > 0 and keypoints_list[17][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[15][0]), int(keypoints_list[15][1])),
                 (int(keypoints_list[17][0]), int(keypoints_list[17][1])), (255, 0, 84), 8)
    if keypoints_list[14][2] > 0 and keypoints_list[16][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[14][0]), int(keypoints_list[14][1])),
                 (int(keypoints_list[16][0]), int(keypoints_list[16][1])), (255, 0, 255), 8)
    if keypoints_list[0][2] > 0 and keypoints_list[15][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[0][0]), int(keypoints_list[0][1])),
                 (int(keypoints_list[15][0]), int(keypoints_list[15][1])), (255, 0, 167), 8)
    if keypoints_list[0][2] > 0 and keypoints_list[14][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[0][0]), int(keypoints_list[0][1])),
                 (int(keypoints_list[14][0]), int(keypoints_list[14][1])), (167, 0, 255), 8)
    if keypoints_list[0][2] > 0 and keypoints_list[1][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[0][0]), int(keypoints_list[0][1])),
                 (int(keypoints_list[1][0]), int(keypoints_list[1][1])), (0, 0, 255), 8)
    if keypoints_list[1][2] > 0 and keypoints_list[2][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[1][0]), int(keypoints_list[1][1])),
                 (int(keypoints_list[2][0]), int(keypoints_list[2][1])), (0, 84, 255), 8)
    if keypoints_list[1][2] > 0 and keypoints_list[5][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[1][0]), int(keypoints_list[1][1])),
                 (int(keypoints_list[5][0]), int(keypoints_list[5][1])), (0, 255, 167), 8)
    if keypoints_list[2][2] > 0 and keypoints_list[3][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[2][0]), int(keypoints_list[2][1])),
                 (int(keypoints_list[3][0]), int(keypoints_list[3][1])), (0, 167, 255), 8)
    if keypoints_list[3][2] > 0 and keypoints_list[4][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[3][0]), int(keypoints_list[3][1])),
                 (int(keypoints_list[4][0]), int(keypoints_list[4][1])), (0, 255, 255), 8)
    if keypoints_list[5][2] > 0 and keypoints_list[6][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[5][0]), int(keypoints_list[5][1])),
                 (int(keypoints_list[6][0]), int(keypoints_list[6][1])), (0, 255, 84), 8)
    if keypoints_list[6][2] > 0 and keypoints_list[7][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[6][0]), int(keypoints_list[6][1])),
                 (int(keypoints_list[7][0]), int(keypoints_list[7][1])), (0, 255, 0), 8)
    if keypoints_list[1][2] > 0 and keypoints_list[8][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[1][0]), int(keypoints_list[1][1])),
                 (int(keypoints_list[8][0]), int(keypoints_list[8][1])), (84, 255, 0), 8)
    if keypoints_list[1][2] > 0 and keypoints_list[11][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[1][0]), int(keypoints_list[1][1])),
                 (int(keypoints_list[11][0]), int(keypoints_list[11][1])), (255, 167, 0), 8)
    if keypoints_list[8][2] > 0 and keypoints_list[9][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[8][0]), int(keypoints_list[8][1])),
                 (int(keypoints_list[9][0]), int(keypoints_list[9][1])), (167, 255, 0), 8)
    if keypoints_list[9][2] > 0 and keypoints_list[10][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[9][0]), int(keypoints_list[9][1])),
                 (int(keypoints_list[10][0]), int(keypoints_list[10][1])), (255, 255, 0), 8)
    if keypoints_list[11][2] > 0 and keypoints_list[12][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[11][0]), int(keypoints_list[11][1])),
                 (int(keypoints_list[12][0]), int(keypoints_list[12][1])), (255, 84, 0), 8)
    if keypoints_list[12][2] > 0 and keypoints_list[13][2] > 0:
        cv2.line(img_sk, (int(keypoints_list[12][0]), int(keypoints_list[12][1])),
                 (int(keypoints_list[13][0]), int(keypoints_list[13][1])), (252, 0, 0), 8)

if __name__=='__main__':
    list = json_list()
    read_json(list)