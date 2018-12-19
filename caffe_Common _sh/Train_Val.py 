#-*- coding: UTF-8 -*-
# Train_Val.py           # 制作train.txt val.txt数据集
import random       # 随机数模块

#############################################
# 针对数据集,制作训练验证样本,并进行相应的扩充
for i in range(12):
    w_val = open('./train_val/val/%03d.txt'%i,'w')
    w_train = open('./train_val/train//%03d.txt'%i,'w')
    with open('./train_val//%03d.txt'%i,'r') as f:
        lines = f.readlines()
        for line in lines:
            a = random.randint(0,9)
            if a == 9:
                if i!=4 and i!=5:
                    w_val.write(line)
                    w_val.write(line)
                else:
                    w_val.write(line)
            else:
                if i!=4 and i!=5:
                    w_train.write(line)
                    w_train.write(line)
                else:
                    w_train.write(line)
    w_val.close()
    w_train.close()
    f.close()

val = open('./train_val/val/val.txt', 'w')
train = open('./train_val/train/train.txt', 'w')
for i in range(12):     # 查询数据长度
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

