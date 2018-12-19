#-*- coding: UTF-8 -*-
# Creat_Folders.py           # 根据输入格式,创建输出文件夹
import os

##################################################################
# 如果输出文件夹不存在,创建文件夹,
def Creat_Folders_(input_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = os.listdir(input_dir)
    out_dir_ = {}
    for in_dir in dataset:
        out_dir = os.path.join(output_dir,in_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_dir_[in_dir] = out_dir

    print ("Creat_Folders.py has created %s Folders successfully" % output_dir.split('/')[-1])
    return out_dir_

##################################################################
# 如果输出文件夹不存在,创建文件夹,返回对应标签的存储路径
def Creat_Folders_dir(input_dir,output_dir,Category_labels):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = os.listdir(input_dir)
    out_dir_ = {}
    for in_dir in dataset:
        out_dir = os.path.join(output_dir,Category_labels[in_dir])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_dir_[in_dir] = out_dir

    print ("Creat_Folders.py has created %s Folders successfully" % output_dir.split('/')[-1])
    return out_dir_

##################################################################
# 删除文件夹
def Remove_Folders(output_dir):
    for file_folder in os.listdir(output_dir):  # 加载文件下所有文件
        for filename in os.listdir(os.path.join(output_dir,file_folder)):
            os.remove(os.path.join(output_dir, file_folder,filename))       # 删除文件
        os.rmdir(os.path.join(output_dir, file_folder))  # 删除文件夹
    os.rmdir(output_dir)
    print ("%s has been removed!!" % output_dir.split('/')[-1])
    return

##################################################################
# 删除文件
def Remove_Files(output_dir,ends_with='.jpg'):
    for file_folder in os.listdir(output_dir):  # 加载文件下所有文件
        for filename in os.listdir(os.path.join(output_dir,file_folder)):
            if filename.endswith(ends_with):
                os.remove(os.path.join(output_dir, file_folder,filename))
        print ("%s's files has been done!!" % file_folder)
    return
