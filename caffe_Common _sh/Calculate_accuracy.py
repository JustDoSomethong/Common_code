#!/usr/bin/env python
# coding=utf-8
import numpy as np

attr_file = '/home/pro/Project/facial_attributes/align_test_attr.txt'
resu_file = '/home/pro/Project/facial_attributes/result_130000.txt'
text = ["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald", "Bangs","Big_Lips","Big_Nose",
            "Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee",
            "Gray_Hair", "Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard",
            "Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair",
            "Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]

with open(attr_file, 'r') as fin:
    attr_lines = fin.readlines()
fin.close()
with open(resu_file, 'r') as fin:
    resu_lines = fin.readlines()
fin.close()

acc = np.zeros(40)
acc_fin = 0
for line in range(len(attr_lines)):
    name_attr = attr_lines[line].strip().replace('  ', ' ').split(' ')
    name_resu = resu_lines[line].strip().replace('   ', '  ').split('  ')
    attr = [int(a) for a in name_attr[1:]]
    resu = [int(a) for a in name_resu[1:]]
    if name_attr[0] == name_resu[0]:
       for i in range(40):
           if attr[i] == resu[i]:
               acc[i] = acc[i] + 1
               acc_fin = acc_fin +1;
           #print text[i].rjust(20) + " : \t",attr[i]
print acc/len(attr_lines)
print float(acc_fin)/len(attr_lines)/40