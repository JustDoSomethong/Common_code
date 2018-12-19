#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/lmdb
DATA=/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/lmdb
TOOLS=/home/pro/caffe-ssd/build/tools

rm -rf $EXAMPLE/imagenet_mean.binaryproto

$TOOLS/compute_image_mean \
	$DATA/train_lmdb \
  	$EXAMPLE/imagenet_mean.binaryproto

echo "Done."
