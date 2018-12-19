#-*- coding: UTF-8 -*-
# Redundancy.py           # 根据输入文件夹的内容,去除文件中子文件的冗余图像
import os
import cv2

##################################################################
# 将去冗余图像保存,保存结果为原名
def Redundancy_for_folders(input_dir,out_dir_lable,image_format,trim_format='%06d'):
    print ("Redundancy begin!!")
    dataset = os.listdir(input_dir)
    for in_dir in dataset:
        res_dir = Redundancy(os.path.join(input_dir,in_dir),image_format)
        for i in range(len(res_dir)):
            res_name = str('finnal_img_'+ "%06d" % i + '.jpg')
            cv2.imwrite(os.path.join(out_dir_lable[in_dir],res_name),res_dir[i])
    print ("Redundancy done!!")
        # print out_dir_lable[in_dir]

##################################################################
# 去子文件夹中冗余图像,返回剩余图像list
def Redundancy(data_in_folder,image_format):
    res_dir = []
    tmp_dir = []
    find = 0
    # print data_in_folder
    for filename in os.listdir(data_in_folder):
        if filename.endswith(image_format):
            img = cv2.imread(os.path.join(data_in_folder,filename))
            # img = cv2.resize(img,(300,300),interpolation=cv2.INTER_CUBIC)
            tmp_dir.append(img)
            for tmp in res_dir:     # 找到相同元素
                if img.shape == tmp.shape:
                    res = cv2.subtract(img,tmp)
                    #print type(res),res.shape,sum(sum(sum(res)))
                    if sum(sum(sum(abs(res)))) == 0:
                        find = 1
            if find == 0:
                res_dir.append(img)
            find = 0
            #print os.path.join(data_in_folder,filename)
    print ("The Folder %s has been finished" % data_in_folder.split('/')[-1], ", %d " % len(tmp_dir), "to %d" % len(res_dir))
    return res_dir
