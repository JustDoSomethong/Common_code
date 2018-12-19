%%
%该代码根据已生成的xml，制作VOC2007数据集中的trainval.txt;train.txt;test.txt和val.txt
%trainval占总数据集的90%，test占总数据集的10%；train占trainval的90%，val占trainval的10%；
%上面所占百分比可根据自己的数据集修改
%注意修改下面两个路径
xmlfilepath='/home/pro/py-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations/';
txtsavepath='/home/pro/py-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/';

xmlfile=dir(xmlfilepath);
numOfxml=length(xmlfile)-2;%减去.和..  总的数据集大小

trainval=sort(randperm(numOfxml,floor(numOfxml*0.9)));%trainval为数据集的90%
test=sort(setdiff(1:numOfxml,trainval));%test为剩余10%

trainvalsize=length(trainval);%trainval的大小
train=sort(trainval(randperm(trainvalsize,floor(trainvalsize*0.9))));
val=sort(setdiff(trainval,train));

ftrainval=fopen([txtsavepath 'trainval.txt'],'w');
ftest=fopen([txtsavepath 'test.txt'],'w');
ftrain=fopen([txtsavepath 'train.txt'],'w');
fval=fopen([txtsavepath 'val.txt'],'w');

for i=1:numOfxml
    if ismember(i,trainval)
        fprintf(ftrainval,'%s\n',xmlfile(i+2).name(1:end-4));
        if ismember(i,train)
            fprintf(ftrain,'%s\n',xmlfile(i+2).name(1:end-4));
        else
            fprintf(fval,'%s\n',xmlfile(i+2).name(1:end-4));
        end
    else
        fprintf(ftest,'%s\n',xmlfile(i+2).name(1:end-4));
    end
end
fclose(ftrainval);
fclose(ftrain);
fclose(fval);
fclose(ftest);