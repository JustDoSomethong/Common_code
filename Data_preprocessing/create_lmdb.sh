set -e 

EXAMPLE=/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/lmdb #生成模型训练数据的路径
PIC_DATA=/ #训练和验证图片所在文件夹路径
TRAIN_DATA=/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/train_val/train/train_out.txt  #训练集txt文件路径
VAL_DATA=/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/train_val/val/val.txt  #验证集txt文件路径
TOOLS=/home/pro/caffe-ssd/build/tools #caffe的工具库，不用更改  

BACKEND="lmdb"

mkdir -p $EXAMPLE

echo "Creating train_${BACKEND}..."

rm -rf $EXAMPLE/train_${BACKEND} 
rm -rf $EXAMPLE/test_${BACKEND}  #删除已存在的lmdb格式文件，

#$TOOLS/convert_imageset --resize_height=256 --resize_width=256 --shuffle $PIC_DATA $TRAIN_DATA $EXAMPLE/train_${BACKEND} --backend=${BACKEND}
$TOOLS/convert_imageset --resize_height=224 --resize_width=224 $PIC_DATA $TRAIN_DATA $EXAMPLE/train_${BACKEND} --backend=${BACKEND}

echo "Creating test_${BACKEND}..."
#$TOOLS/convert_imageset --resize_height=256 --resize_width=256 --shuffle $PIC_DATA $VAL_DATA $EXAMPLE/test_${BACKEND} --backend=${BACKEND}
$TOOLS/convert_imageset --resize_height=224 --resize_width=224 $PIC_DATA $VAL_DATA $EXAMPLE/test_${BACKEND} --backend=${BACKEND}
echo "Done."
