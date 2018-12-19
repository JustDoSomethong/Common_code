#-*- coding: UTF-8 -*-
# Train_Val.py           # 制作train.txt val.txt数据集
import random       # 随机数模块
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--category_num',type=int,default = 13,help='类别种类数目')
# parser.add_argument('--Augmentation_num',type=int,default = 5,help='数据增广数目')
# parser.add_argument('--train_label_dir',type=str,default='/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/train_val/train/train.txt',help='label标签按大小排序的train.txt')
# parser.add_argument('--train_output_dir',type=str,default='/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/train_val/train/train_out.txt',help='train_out.txt保存路径')
# args = parser.parse_args()

#############################################
# 针对数据集,制作训练验证样本
def Resample_for_Trian_Val(category_num):
    for i in range(category_num):
        w_val = open('./train_val/val/%03d.txt'%i,'w')
        w_train = open('./train_val/train/%03d.txt'%i,'w')
        with open('./train_val/%03d.txt'%i,'r') as f:
            lines = f.readlines()
            for line in lines:
                a = random.randint(0,9)
                if a == 9:
                    w_val.write(line)
                else:
                    w_train.write(line)
        w_val.close()
        w_train.close()
        f.close()

    val = open('./train_val/val/val.txt', 'w')
    train = open('./train_val/train/train.txt', 'w')
    for i in range(category_num):     # 查询数据长度
        w_val = open('./train_val/val/%03d.txt' % i, 'r')
        w_train = open('./train_val/train/%03d.txt' % i, 'r')
        lines = w_val.readlines()
        print '%03d_val' %i, len(lines)
        for line in lines:
            val.write(line[0:-1]+' %d\n' %i)
        lines = w_train.readlines()
        for line in lines:
            train.write(line[0:-1]+' %d\n' %i)
        print '%03d_train' % i, len(lines)

#############################################
# 针对数据集,制作训练验证样本
def Resample_for_Trian_Val_for_Data_Augmentation(category_num,Augmentation_num):
    for i in range(category_num):
        w_val = open('./train_val/val/%03d.txt' % i, 'w')
        w_train = open('./train_val/train/%03d.txt' % i, 'w')
        with open('./train_val/%03d.txt' % i, 'r') as f:
            lines = f.readlines()
            for line in lines:
                a = random.randint(0, 9)
                if a == 9:
                    w_val.write(line)
                else:
                    w_train.write(line)
        w_val.close()
        w_train.close()
        f.close()

    val = open('./train_val/val/val.txt', 'w')
    train = open('./train_val/train/train.txt', 'w')
    for i in range(category_num):  # 查询数据长度
        w_val = open('./train_val/val/%03d.txt' % i, 'r')
        w_train = open('./train_val/train/%03d.txt' % i, 'r')
        lines = w_val.readlines()
        # print '%03d_val' % i, len(lines)
        for line in lines:
            val.write(line[0:-1] + ' %d\n' % i)
            for j in range(1,Augmentation_num):
                val.write(line[0:-5] + '_c%03d.jpg' % j + ' %d\n' % i)
        lines = w_train.readlines()
        for line in lines:
            train.write(line[0:-1] + ' %d\n' % i)
            for j in range(1,Augmentation_num):
                train.write(line[0:-5] + '_c%03d.jpg' % j + ' %d\n' % i)
        # print '%03d_train' % i, len(lines)

#############################################
# 针对数据集,制作训练验证样本,运用Label Shuffing,数据均衡
def Label_Shuffing_for_Trian_Val(category_num,train_label_dir,train_output_dir):
    dicts={}
    f = open(train_label_dir,'r')
    lines = f.readlines()
    for line in lines:
        img_dir = line.split()[0]
        label = int(line.split()[1])
        dicts[img_dir] = label
    dicts = sorted(dicts.items(), key=lambda item: item[1])     # 按照label从小到大排序
    f.close()

    # 统计每一类label的数目
    counts = {}
    new_dicts = []
    for i in range(category_num):
        counts[i] = 0
    for line in dicts:
        if line[1] > category_num - 1:
            continue
        line = list(line)
        line.append(counts[line[1]])
        counts[line[1]] += 1
        new_dicts.append(line)
    print counts

    # 把原列表按照每一类分成各个block并形成新列表
    tab = []
    origin_index = 0
    for i in range(category_num):
        block = []
        for j in range(counts[i]):
            block.append(new_dicts[origin_index])
            origin_index += 1
        # print block
        tab.append(block)
    # print tab

    nums = []  # 找到数目最多的label类别,从大到校排序
    for key in counts:
        nums.append(counts[key])
    nums.sort(reverse=True)
    # print nums

    lists = []  # 形成随机label序列
    for i in range(nums[0]):
        lists.append(i)
    # print lists

    all_index = []
    for i in range(category_num):
        random.shuffle(lists)
        # print lists
        lists_res = [j % counts[i] for j in lists]
        all_index.append(lists_res)
        # print lists_res
    # print all_index[10]

    f = open(train_output_dir, 'w')
    shuffle_labels = []
    index = 0
    for line in all_index:
        for i in line:
            shuffle_labels.append(tab[index][i])
        index += 1
    # print shuffle_labels
    random.shuffle(shuffle_labels)
    id = 0
    for line in shuffle_labels:
        # print line
        f.write(line[0]+' '+str(line[1]))
        # f.write(str(id) + '\t' + str(line[1]) + '\t' + line[0])
        f.write('\n')
        id += 1
    f.close()

#############################################
# 拆分数据集trainval,制作训练验证数据集train,val
def Transfor_Trianval_to_train_val():
    w_val = open('./SSD_data/ImageSets/Main/val.txt','w')
    w_train = open('./SSD_data/ImageSets/Main/train.txt','w')
    with open('./SSD_data/ImageSets/Main/trainval.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            a = random.randint(0,9)
            if a == 9:
                w_val.write(line)
            else:
                w_train.write(line)
    w_val.close()
    w_train.close()
    f.close()

if __name__ == '__main__':
    Transfor_Trianval_to_train_val()
