import sys
# add your caffe/python path
sys.path.insert(0, "/home/ling/YH/caffe-ssd/python")
import caffe
import sys
caffe.set_mode_cpu()
import numpy as np
from numpy import prod, sum

def print_net_parameters_flops (deploy_file):
    net = caffe.Net(deploy_file, caffe.TEST)
    print ("Total number of parameters: %.2fMB" %(sum([prod(v[0].data.shape) for k, v in net.params.items()])/1e6))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ('Usage:')
        print ('python compute_flops_params.py  deploy.prototxt')
        exit()
    deploy_file = sys.argv[1]
    print_net_parameters_flops(deploy_file)
