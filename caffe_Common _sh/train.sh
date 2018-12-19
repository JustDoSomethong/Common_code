#!/usr/bin/env sh
set -e

/home/pro/caffe-master/build/tools/caffe train \
	--gpu=0 \
	--solver=/home/pro/Project/hdf5/solver.prototxt \
	--weights=/home/pro/Project/hdf5/VGG_ILSVRC_16_layers.caffemodel 2>&1| tee /home/pro/Project/hdf5/out.log $@
