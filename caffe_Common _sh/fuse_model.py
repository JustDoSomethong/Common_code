# coding=utf-8
# caffe常用sh/fuse_model.py		文件为py程序，融合模型，生成融合后的权重
# 代码将将训练好的两个模型，如 VGG16 进行融合。主要解决两个模型中的层的名字是相同的，而对应不同的权重；同时将两个训练好的权重读入一个融合好的模型。
# 第一步，准备网络模型以及预训练模型
# 第二步，制作两个网络模型以及对应的预训练模型，可以改名为 even/odd
# 第三步，执行代码 even_odd_prototxt 改变模型Prototxt，注意，data层会出错，需要主动修改
# 第四步，将两个模型融合，并且定义融合层和数据层
# 第五步，融合模型参数

import re
import sys
sys.path.append('/home/ling/YH/caffe-master/python')
import caffe

############################################################
# 该代码改变模型Prototxt，注意，data层会出错，需要主动修改
def even_odd_prototxt(input_prototxt,out_prototxt,even_or_odd):
    layer_name_regex = re.compile('name:\s*"(.*?)"')
    # lr_mult_regex = re.compile('lr_mult:\s*\d+\.*\d*')

    with open(input_prototxt, 'r') as fr, open(out_prototxt, 'w') as fw:
        prototxt = fr.read()
        layer_names = list(layer_name_regex.findall(prototxt))
        for layer_name in layer_names:
            prototxt = prototxt.replace(layer_name, '{0}_{1}'.format(even_or_odd, layer_name))

        # lr_mult_statements = set(lr_mult_regex.findall(prototxt))
        # for lr_mult_statement in lr_mult_statements:
        #     prototxt = prototxt.replace(lr_mult_statement, 'lr_mult: 0')

        fw.write(prototxt)

############################################################
def fuse_even_model():
    fusion_net = caffe.Net('./fuse_even_odd_train_val.prototxt', caffe.TEST)
    net = caffe.Net('even_train_val.prototxt', 'VGG_even.caffemodel', caffe.TEST)
    for layer_name, param in net.params.iteritems():
        n_params = len(param)
        if '{}_{}'.format('even', layer_name) in fusion_net.params:
            print(layer_name,'{}_{}'.format('even', layer_name))
            try:
                for i in range(n_params):
                    fusion_net.params['{}_{}'.format('even', layer_name)][i].data[...] = param[i].data[...]
            except Exception as e:
                print(e)
    fusion_net.save('init_fusion.caffemodel')

    # model_list = [
    #     ('even', 'even_train_val.prototxt', 'VGG_even.caffemodel'),
    #     ('odd', 'odd_train_val.prototxt', 'VGG_odd.caffemodel')
    # ]
    # for prefix, model_def, model_weight in model_list:
    #     net = caffe.Net(model_def, model_weight, caffe.TEST)
    #     for layer_name, param in net.params.iteritems():
    #         n_params = len(param)
    #         try:
    #             for i in range(n_params):
    #                 fusion_net.params['{}/{}'.format(prefix, layer_name)][i].data[...] = param[i].data[...]
    #         except Exception as e:
    #             print(e)
    #
    # fusion_net.save('init_fusion.caffemodel')

############################################################
def fuse_odd_model():
    fusion_net = caffe.Net('./fuse_even_odd_train_val.prototxt', 'init_fusion.caffemodel', caffe.TEST)
    net = caffe.Net('odd_train_val.prototxt', 'VGG_odd.caffemodel', caffe.TEST)
    for layer_name, param in net.params.iteritems():
        n_params = len(param)
        if '{}_{}'.format('odd', layer_name) in fusion_net.params:
            print(layer_name, '{}_{}'.format('odd', layer_name))
            try:
                for i in range(n_params):
                    fusion_net.params['{}_{}'.format('odd', layer_name)][i].data[...] = param[i].data[...]
            except Exception as e:
                print(e)

    fusion_net.save('init_fusion.caffemodel')

############################################################
if __name__ == '__main__':
    # even_odd_prototxt('./even_train_val.prototxt','./change_even_train_val.prototxt','even')
    # even_odd_prototxt('./odd_train_val.prototxt','./change_odd_train_val.prototxt','odd')
    # fuse_even_model()
    fuse_odd_model()