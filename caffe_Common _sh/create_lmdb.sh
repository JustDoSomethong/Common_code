set -e 

EXAMPLE=/media/pro/PRO/Ubuntu/常用代码/数据预处理 #生成模型训练数据的路径
PIC_DATA=/media/pro/PRO/Ubuntu/常用代码/数据预处理/ #训练和验证图片所在文件夹路径
TRAIN_DATA=/media/pro/PRO/Ubuntu/常用代码/数据预处理/train_val/train/train.txt  #训练集txt文件路径
VAL_DATA=/media/pro/PRO/Ubuntu/常用代码/数据预处理/train_val/val/val.txt  #验证集txt文件路径
TOOLS=/home/pro/caffe-master/build/tools #caffe的工具库，不用更改  

BACKEND="lmdb"

echo "Creating train_${BACKEND}..."

rm -rf $EXAMPLE/train_${BACKEND} 
rm -rf $EXAMPLE/test_${BACKEND}  #删除已存在的lmdb格式文件，

$TOOLS/convert_imageset --resize_height=224 --resize_width=224 --shuffle $PIC_DATA $TRAIN_DATA $EXAMPLE/train_${BACKEND} --backend=${BACKEND}

echo "Creating test_${BACKEND}..."

$TOOLS/convert_imageset --resize_height=224 --resize_width=224 --shuffle $PIC_DATA $VAL_DATA $EXAMPLE/test_${BACKEND} --backend=${BACKEND}
echo "Done."
