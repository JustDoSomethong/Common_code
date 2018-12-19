#!/usr/bin/env sh
set -e

/home/pro/caffe-master/build/tools/caffe train \
    --gpu=0 \
    --solver=/home/pro/Project/hdf5/solver.prototxt \
    --snapshot=/home/pro/Project/hdf5/_iter_100000.solverstate 2>&1| tee /home/pro/Project/hdf5/out.log $@
