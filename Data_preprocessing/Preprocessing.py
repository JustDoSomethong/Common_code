#-*- coding: UTF-8 -*-
# Preprocessing.py      # 完整的图像预处理
import argparse,os
from Creat_Folders import Creat_Folders_dir,Remove_Folders
from Redundancy import Redundancy_for_folders
from Rename import Rename_for_Category_labels_
from Train_Val import Resample_for_Trian_Val_for_Data_Augmentation,Label_Shuffing_for_Trian_Val
from Resize_Crop import Resize_Crop_for_Images
from Data_Augmentation import Data_Augmentation_

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',type=str,default='./检测数据结果',help='输入文件夹位置')
parser.add_argument('--output_dir',type=str,default='./data',help='输出文件夹保存位置')
parser.add_argument('--tmp_dir',type=str,default='./tmp',help='中间结果保存位置')
parser.add_argument('--crop_size',type=str,default='224',help='中心裁剪保存结果')
parser.add_argument('--image_format',type=str,default='jpg',help='图片默认后缀')
parser.add_argument('--trim_format',type=str,default='%06d')      # 保存名字格式
parser.add_argument('--category_num',type=int,default = 13,help='类别种类数目')
parser.add_argument('--Augmentation_num',type=int,default = 0,help='数据增广数目')
parser.add_argument('--train_label_dir',type=str,default='/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/train_val/train/train.txt',help='label标签按大小排序的train.txt')
parser.add_argument('--train_output_dir',type=str,default='/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/train_val/train/train_out.txt',help='train_out.txt保存路径')
args = parser.parse_args()

 # 构建字典,对应类别标签
Category_labels ={ '吃东西':'000',
                   '喝水':'001',
                   '打电话':'002',
                   '看手机':'003',
                   '看书':'004',
                   '头疼':'005',
                   '牙疼':'006',
                   '胸疼':'007',
                   '腰疼':'008',
                   '腹痛':'009',
                   '跌倒':'010',
                   '脸部假动作':'011',
                   '坐立正常':'012'
                   }

#Category_labels ={'000':'000',
#                  '001':'001',
#                  '002':'002',
#                  '003':'003',
#                  '004':'004',
#                  '005':'005',
#                  '006':'006',
#                  '007':'007',
#                  '008':'008',
#                  '009':'009',
#                  '010':'010',
#                  '011':'011',
#                  '012':'012'
#                  }

if __name__ == '__main__':
    out_dir_ = Creat_Folders_dir(args.input_dir, args.output_dir,Category_labels)   # 根据输入格式,创建输出文件夹
    tmp_dir_ = Creat_Folders_dir(args.input_dir, args.tmp_dir, Category_labels)  # 根据输入格式,创建中间结果文件夹
    Redundancy_for_folders(args.input_dir,tmp_dir_,args.image_format,trim_format = args.trim_format)    # 图像去冗余,保存去冗余后图像结果,保存置中间结果中
    # Resize_Crop_for_Images(args.tmp_dir,image_crop_size=args.crop_size)	# resize图像大小
    Rename_for_Category_labels_(args.tmp_dir, args.output_dir, args.image_format, args.trim_format)   # 文件重命名
    Remove_Folders(args.tmp_dir)    # 删除中间保存结果文件
    # os.system('./txt.sh')       # 根据txt.sh文件制作每个类别的txt
    # Resample_for_Trian_Val_for_Data_Augmentation(args.category_num,args.Augmentation_num)   # 制作train.txt val.txt数据集
    # Label_Shuffing_for_Trian_Val(args.category_num,args.train_label_dir,args.train_output_dir)  # 制作train.txt val.txt数据集
    # # Data_Augmentation_(args.output_dir, args.image_format)  # 图像增强,颜色抖动,一定在之后运行
    # os.system('./create_lmdb.sh')   # 文件产生lmdb文件,输入为txt文件,存有图像的路径以及标签(convert_imageset)
    # os.system('./make_imagenet_mean.sh')    # 文件获取图像的均值,输入为lmdb文件train_lmdb(compute_image_mean)
