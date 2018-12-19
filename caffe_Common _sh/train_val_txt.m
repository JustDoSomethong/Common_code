clc
close all
clear all
filepath = '/home/pro/文档/caffe数据库和模型/数据库/food/images/apple_pie'; %%获取图片地址
filename = dir([filepath,'/','*.jpg']); %%运用dir函数读取当前目录下的.bmp文件,并返回结构体,用[]将地址组合在一起
n = 0.9*length(filename); %%获取结构体的长度
m = length(filename); %%获取结构体的长度
fid_w_train = fopen('train.txt','wt');
fid_w_val = fopen('val.txt','wt');
zero = num2str(0);  %%将数值转化为字符串
for i=1:n
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['apple_pie/',name];
    fprintf(fid_w_train,'%s ',name);
    fprintf(fid_w_train,'%s\n',zero);
end 
for i=n+1:m
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['apple_pie/',name];
    fprintf(fid_w_val,'%s ',name);
    fprintf(fid_w_val,'%s\n',zero);
end

filepath = '/home/pro/文档/caffe数据库和模型/数据库/food/images/baby_back_ribs'; %%获取图片地址
filename = dir([filepath,'/','*.jpg']); %%运用dir函数读取当前目录下的.bmp文件,并返回结构体,用[]将地址组合在一起
n = 0.9*length(filename); %%获取结构体的长度
m = length(filename); %%获取结构体的长度
zero = num2str(1);  %%将数值转化为字符串
for i=1:n
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['baby_back_ribs/',name];
    fprintf(fid_w_train,'%s ',name);
    fprintf(fid_w_train,'%s\n',zero);
end 
for i=n+1:m
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['baby_back_ribs/',name];
    fprintf(fid_w_val,'%s ',name);
    fprintf(fid_w_val,'%s\n',zero);
end

filepath = '/home/pro/文档/caffe数据库和模型/数据库/food/images/baklava'; %%获取图片地址
filename = dir([filepath,'/','*.jpg']); %%运用dir函数读取当前目录下的.bmp文件,并返回结构体,用[]将地址组合在一起
n = 0.9*length(filename); %%获取结构体的长度
m = length(filename); %%获取结构体的长度
zero = num2str(2);  %%将数值转化为字符串
for i=1:n
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['baklava/',name];
    fprintf(fid_w_train,'%s ',name);
    fprintf(fid_w_train,'%s\n',zero);
end 
for i=n+1:m
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['baklava/',name];
    fprintf(fid_w_val,'%s ',name);
    fprintf(fid_w_val,'%s\n',zero);
end

filepath = '/home/pro/文档/caffe数据库和模型/数据库/food/images/beef_carpaccio'; %%获取图片地址
filename = dir([filepath,'/','*.jpg']); %%运用dir函数读取当前目录下的.bmp文件,并返回结构体,用[]将地址组合在一起
n = 0.9*length(filename); %%获取结构体的长度
m = length(filename); %%获取结构体的长度
zero = num2str(3);  %%将数值转化为字符串
for i=1:n
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['beef_carpaccio/',name];
    fprintf(fid_w_train,'%s ',name);
    fprintf(fid_w_train,'%s\n',zero);
end 
for i=n+1:m
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['beef_carpaccio/',name];
    fprintf(fid_w_val,'%s ',name);
    fprintf(fid_w_val,'%s\n',zero);
end

filepath = '/home/pro/文档/caffe数据库和模型/数据库/food/images/beef_tartare'; %%获取图片地址
filename = dir([filepath,'/','*.jpg']); %%运用dir函数读取当前目录下的.bmp文件,并返回结构体,用[]将地址组合在一起
n = 0.9*length(filename); %%获取结构体的长度
m = length(filename); %%获取结构体的长度
zero = num2str(4);  %%将数值转化为字符串
for i=1:n
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['beef_tartare/',name];
    fprintf(fid_w_train,'%s ',name);
    fprintf(fid_w_train,'%s\n',zero);
end 
for i=n+1:m
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['beef_tartare/',name];
    fprintf(fid_w_val,'%s ',name);
    fprintf(fid_w_val,'%s\n',zero);
end

filepath = '/home/pro/文档/caffe数据库和模型/数据库/food/images/beet_salad'; %%获取图片地址
filename = dir([filepath,'/','*.jpg']); %%运用dir函数读取当前目录下的.bmp文件,并返回结构体,用[]将地址组合在一起
n = 0.9*length(filename); %%获取结构体的长度
m = length(filename); %%获取结构体的长度
zero = num2str(5);  %%将数值转化为字符串
for i=1:n
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['beet_salad/',name];
    fprintf(fid_w_train,'%s ',name);
    fprintf(fid_w_train,'%s\n',zero);
end 
for i=n+1:m
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['beet_salad/',name];
    fprintf(fid_w_val,'%s ',name);
    fprintf(fid_w_val,'%s\n',zero);
end

filepath = '/home/pro/文档/caffe数据库和模型/数据库/food/images/beignets'; %%获取图片地址
filename = dir([filepath,'/','*.jpg']); %%运用dir函数读取当前目录下的.bmp文件,并返回结构体,用[]将地址组合在一起
n = 0.9*length(filename); %%获取结构体的长度
m = length(filename); %%获取结构体的长度
zero = num2str(6);  %%将数值转化为字符串
for i=1:n
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['beignets/',name];
    fprintf(fid_w_train,'%s ',name);
    fprintf(fid_w_train,'%s\n',zero);
end 
for i=n+1:m
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['beignets/',name];
    fprintf(fid_w_val,'%s ',name);
    fprintf(fid_w_val,'%s\n',zero);
end

filepath = '/home/pro/文档/caffe数据库和模型/数据库/food/images/bibimbap'; %%获取图片地址
filename = dir([filepath,'/','*.jpg']); %%运用dir函数读取当前目录下的.bmp文件,并返回结构体,用[]将地址组合在一起
n = 0.9*length(filename); %%获取结构体的长度
m = length(filename); %%获取结构体的长度
zero = num2str(7);  %%将数值转化为字符串
for i=1:n
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['bibimbap/',name];
    fprintf(fid_w_train,'%s ',name);
    fprintf(fid_w_train,'%s\n',zero);
end 
for i=n+1:m
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['bibimbap/',name];
    fprintf(fid_w_val,'%s ',name);
    fprintf(fid_w_val,'%s\n',zero);
end

filepath = '/home/pro/文档/caffe数据库和模型/数据库/food/images/bread_pudding'; %%获取图片地址
filename = dir([filepath,'/','*.jpg']); %%运用dir函数读取当前目录下的.bmp文件,并返回结构体,用[]将地址组合在一起
n = 0.9*length(filename); %%获取结构体的长度
m = length(filename); %%获取结构体的长度
zero = num2str(8);  %%将数值转化为字符串
for i=1:n
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['bread_pudding/',name];
    fprintf(fid_w_train,'%s ',name);
    fprintf(fid_w_train,'%s\n',zero);
end 
for i=n+1:m
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['bread_pudding/',name];
    fprintf(fid_w_val,'%s ',name);
    fprintf(fid_w_val,'%s\n',zero);
end

filepath = '/home/pro/文档/caffe数据库和模型/数据库/food/images/breakfast_burrito'; %%获取图片地址
filename = dir([filepath,'/','*.jpg']); %%运用dir函数读取当前目录下的.bmp文件,并返回结构体,用[]将地址组合在一起
n = 0.9*length(filename); %%获取结构体的长度
m = length(filename); %%获取结构体的长度
zero = num2str(9);  %%将数值转化为字符串
for i=1:n
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['breakfast_burrito/',name];
    fprintf(fid_w_train,'%s ',name);
    fprintf(fid_w_train,'%s\n',zero);
end 
for i=n+1:m
    name = filename(i).name; %%获取结构体中的图像的名字
    name = ['breakfast_burrito/',name];
    fprintf(fid_w_val,'%s ',name);
    fprintf(fid_w_val,'%s\n',zero);
end

