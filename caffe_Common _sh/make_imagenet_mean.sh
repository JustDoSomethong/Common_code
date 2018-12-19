#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/pro/hdf5
DATA=/home/pro/hdf5
TOOLS=build/tools

rm -rf $EXAMPLE/imagenet_mean.binaryproto

$TOOLS/compute_image_mean \
	$DATA/train_lmdb \
  	$EXAMPLE/imagenet_mean.binaryproto

echo "Done."
