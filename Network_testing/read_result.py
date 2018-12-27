#coding=utf-8
# read_result.py 	# 解析 result.log 文件,即对分类输出进行解析
import os

result_log = './result.log'
val_txt = './val_out.txt'
val_result = './val_result.txt'
img_dir = '../Data_preprocessing/shiyandata/Orginal_data_2.0'
res_dir = '../Data_preprocessing/shiyandata/result_crop_0.868304'
##############################################################################
# 读取输出的log文件,进行解析,获得标签labe以及网络测试结果
def read_result_log():
    file = open(result_log,'r')
    lines = file.readlines()
    result_list = []
    single_list = []
    result_map = {}
    for line_number,line in enumerate(lines):
        if (line_number)%13 ==0:
            result_map['lable'] = int(line.split()[-1])
        elif (line_number)%13 != 12 :
            single_list.append(float(line.split()[-1]))
        else:
            single_list.append(float(line.split()[-1]))
            result_map['result'] = single_list.index(max(single_list))
            result_list.append(result_map)
            result_map = {}
            single_list =[]
    file.close()
    return result_list

##############################################################################
# 根据测试数据集val_txt,将图像和标签匹配上,同时保存图像以及对应标签
def read_val(result_list):
    file = open(val_txt,'r')
    f = open(val_result,'w')
    lines = file.readlines()
    for line_number,line in enumerate(lines):
        if result_list[line_number]['lable'] != result_list[line_number]['result']:
            write_str = '{0}\t{1}\t{2}\n'.format(line.split()[0].split('/')[1],result_list[line_number]['lable'],result_list[line_number]['result'])
            val_img_dir = os.path.join(img_dir,line.split()[0].split('/')[1])
            res_img_dir = os.path.join(res_dir,line.split()[0].split('/')[1])
            command = ['cp','%s'%val_img_dir,'%s'%res_img_dir]
            command = ' '.join(command)
            # print(command)
            os.system(command)
            f.write(write_str)
            # print (line_number,line.split()[0].split('/')[1],result_list[line_number])
    f.close()
    file.close()
if __name__ == '__main__':
    result_list = read_result_log()
    read_val(result_list)