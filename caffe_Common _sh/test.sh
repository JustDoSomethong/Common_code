#!/usr/bin/env sh
set -e

/home/kb539/YH/caffe-master/build/tools/caffe test --gpu=0 --model=/home/kb539/YH/work/behavior_recognition/vgg_16/deploy.prototxt --weights=/home/kb539/YH/work/behavior_recognition/vgg_16/output/case_two.caffemodel --iterations=21
