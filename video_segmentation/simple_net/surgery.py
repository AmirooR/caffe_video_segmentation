import caffe
import numpy as np
from net import conv_relu, max_pool, save_net, create_net, conv_vgg
from os import remove
from caffe import layers as L, params

def transplant(new_net, net, suffix=''):
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat

def copy_params(new_net, net, pair_param = []):
    for from_param, to_param in pair_param:
	if from_param in net.params and to_param in new_net.params:
	    for i in range(len(net.params[from_param])):
		if i > (len(new_net.params[to_param]) - 1):
		    print 'dropping', from_param, i
		    break
		if net.params[from_param][i].data.shape != new_net.params[to_param][i].data.shape:
		    print 'coercing', from_param, i, 'from', net.params[from_param][i].data.shape, 'to', new_net.params[to_param][i].data.shape
		else:
		    print 'copying', from_param, ' -> ', to_param, i
		new_net.params[to_param][i].data.flat = net.params[from_param][i].data.flat

def create_dilated_net():
    n = caffe.NetSpec()
    n.im, n.label_1 = L.MemoryData(batch_size=1, channels=3, height=244, width=244, ntop=2)
    conv_vgg(n, n.im, first_layer_pad=100, add_extra_pad = True, use_dilated_net = True)
    # fully conv
    n.fc6, n.relu6 = conv_relu(n.conv5_3, 4096, ks=7, dilation=4)        
    n.fc7, n.relu7 = conv_relu(n.relu6, 4096, ks=1, pad=0)
    n.fc8 = L.Convolution(n.relu7, kernel_size=1, num_output=21)
    return n


def create_vgg_net():
    n = caffe.NetSpec()
    n.im, n.label_1 = L.MemoryData(batch_size=1, channels=3, height=244, width=244, ntop=2)
    conv_vgg(n, n.im, add_extra_pad = False, use_dilated_net = False)
    # fully connected is not necessary
    return n

def simple_net_surgery(datalayer_info, net_info):

    dilated_path = 'dilated_net_tmp.prototxt'
    dilated_weights = './data/dilation8_pascal_voc.caffemodel'
    new_weights = 'simple_net.caffemodel'
    new_path = 'new_net_tmp.prototxt'
    new_net_info = net_info.copy()
    new_net_info['initialize_fcs'] = True 
    dilated_netspec = create_dilated_net()
    new_netspec = create_net('deploy', new_net_info, datalayer_info)
    save_net(dilated_path, str(dilated_netspec.to_proto()))
    save_net(new_path, str(new_netspec.to_proto()))
    dilated_net = caffe.Net(dilated_path, dilated_weights, caffe.TRAIN)
    new_net = caffe.Net(new_path, caffe.TRAIN)
    #transplant vgg-net conv weights
    transplant(new_net, dilated_net, 'c')
    
    #transplant fc6 weights
    #new_net.params['fc6'][0].data[:, -512:][...] = dilated_net.params['fc6'][0].data[:fc6_size]
    #new_net.params['fc6'][1].data[...] = dilated_net.params['fc6'][1].data[:fc6_size]
    
    #transplant fc7 weights
    #new_net.params['fc7'][0].data[...] = dilated_net.params['fc7'][0].data[:fc7_size, :fc6_size]
    #new_net.params['fc7'][1].data[...] = dilated_net.params['fc7'][1].data[:fc7_size]

    new_net.save(new_weights)
    remove(dilated_path)
    remove(new_path)


def fcn8_net_surgery(datalayer_info, net_info):

    fcn8_path = './data/crfasrnn_coco_voc.prototxt'
    fcn8_weights = './data/crfasrnn_coco_voc.caffemodel'
    
    new_weights = 'fcn8_net.caffemodel'
    new_path = 'new_net_tmp.prototxt'
    new_net_info = net_info.copy()
    new_net_info['initialize_fcs'] = True
    new_netspec = create_net('deploy', new_net_info, datalayer_info)
    save_net(new_path, str(new_netspec.to_proto()))
    
    fcn8_net = caffe.Net(fcn8_path, fcn8_weights, caffe.TRAIN)
    new_net = caffe.Net(new_path, caffe.TRAIN)
    
    #transplant vgg-net conv weights
    transplant(new_net, fcn8_net, 'm')
    
    new_net.params['conv1_1m'][0].data[...] = new_net.params['conv1_1m'][0].data * 255
    new_net.params['conv1_1m'][1].data[...] = new_net.params['conv1_1m'][1].data * 255
    
    param_pairs = [('score-fr', 'score_fc'), ('score2', 'upscore2'), ('score-pool4', 'score_pool4'), ('score4', 'upscore_pool4'), ('score-pool3' ,'score_pool3'), ('upsample', 'upscore8')]
    if net_info['fc6_size'] == 4096 and net_info['fc7_size'] == 4096:
	#transplant fc6 weights
	new_net.params['fc6'][0].data[...] = 0
	new_net.params['fc6'][0].data[:, -512:][...] = fcn8_net.params['fc6'][0].data
	new_net.params['fc6'][1].data[...] = fcn8_net.params['fc6'][1].data
	param_pairs.append(('fc7', 'fc7'))

    #transplant the rest
    copy_params(new_net, fcn8_net, pair_param = param_pairs)
    
    new_net.save(new_weights)
    remove(new_path)
    
def deepmask_net_surgery(datalayer_info, net_info):

    vgg_path = 'deepmask_net_tmp.prototxt'
    vgg_weights = './data/fcn32s-heavy-pascal.caffemodel'
    new_weights = 'deepmask_net.caffemodel'
    new_path = 'new_net_tmp.prototxt'
    new_net_info = net_info.copy()
    new_net_info['initialize_fcs'] = True
    vgg_netspec = create_vgg_net()
    new_netspec = create_net('deploy', new_net_info, datalayer_info)
    save_net(vgg_path, str(vgg_netspec.to_proto()))
    save_net(new_path, str(new_netspec.to_proto()))
    vgg_net = caffe.Net(vgg_path, vgg_weights, caffe.TRAIN)
    new_net = caffe.Net(new_path, caffe.TRAIN)
    
    #transplant vgg-net conv weights
    transplant(new_net, vgg_net, 'c')

    new_net.save(new_weights)
    remove(vgg_path)
    remove(new_path)
