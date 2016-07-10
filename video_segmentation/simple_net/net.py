import caffe

from caffe import layers as L, params as P
from caffe.coord_map import coord_map_from_to
import numpy as np

def save_net(net_path, proto):
    with open(net_path, 'w') as f:
        f.write(proto)

def conv_relu(bottom, nout, ks=3, stride=1, pad=0, dilation=1, param_name=None, std=-1):
    conv_param = {'kernel_size':ks}
    if stride != 1:
        conv_param['stride'] = stride
    if not hasattr(pad, '__len__'):
	if pad != 0:
	    conv_param['pad'] = int(pad)
    else:
	if pad[0] != 0:
	    conv_param['pad_w'] = int(pad[0])
	if pad[1] != 0:
	    conv_param['pad_h'] = int(pad[1])
    if dilation != 1:
        conv_param['dilation'] = dilation
    
    conv_param['num_output'] = nout
    
    conv_param['param']= [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    if param_name is not None:
        conv_param['param'] = [dict(name=param_name + '_w', lr_mult=1, decay_mult=1), dict(name=param_name + '_b', lr_mult=2, decay_mult=0)]

    if std > 0:
      conv_param['weight_filler'] = dict(type='gaussian', std=std)
    conv = L.Convolution(bottom, **conv_param)
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def conv_vgg(n, im, suffix='', last_layer_pad=0, first_layer_pad=0, add_extra_pad = True, use_dilated_net = True, add_pool5 = False):
  
    if add_extra_pad:
	pad = 1
    else:
	pad = 0

    conv, relu = conv_relu(im, 64, pad=pad + first_layer_pad, param_name='conv1_1')
    setattr(n, 'conv1_1' + suffix, conv)
    setattr(n, 'relu1_1' + suffix, relu)
    
    conv, relu = conv_relu(relu, 64, pad=pad, param_name='conv1_2')
    setattr(n, 'conv1_2' + suffix, conv)
    setattr(n, 'relu1_2' + suffix, relu)
    
    pool = max_pool(relu)
    setattr(n, 'pool1' + suffix, pool)
    
    
    
    conv, relu = conv_relu(pool, 128, pad=pad, param_name='conv2_1')
    setattr(n, 'conv2_1' + suffix, conv)
    setattr(n, 'relu2_1' + suffix, relu)
    
    conv, relu = conv_relu(relu, 128, pad=pad, param_name='conv2_2')
    setattr(n, 'conv2_2' + suffix, conv)
    setattr(n, 'relu2_2' + suffix, relu)
    
    pool = max_pool(relu)
    setattr(n, 'pool2' + suffix, pool)
    
    
    conv, relu = conv_relu(pool, 256, pad=pad, param_name='conv3_1')
    setattr(n, 'conv3_1' + suffix, conv)
    setattr(n, 'relu3_1' + suffix, relu)
    
    conv, relu = conv_relu(relu, 256, pad=pad, param_name='conv3_2')
    setattr(n, 'conv3_2' + suffix, conv)
    setattr(n, 'relu3_2' + suffix, relu)
    
    conv, relu = conv_relu(relu, 256, pad=pad, param_name='conv3_3')
    setattr(n, 'conv3_3' + suffix, conv)
    setattr(n, 'relu3_3' + suffix, relu)
    pool = max_pool(relu)
    setattr(n, 'pool3' + suffix, pool)
    

    conv, relu = conv_relu(pool, 512, pad=pad, param_name='conv4_1')
    setattr(n, 'conv4_1' + suffix, conv)
    setattr(n, 'relu4_1' + suffix, relu)
    
    conv, relu = conv_relu(relu, 512, pad=pad, param_name='conv4_2')
    setattr(n, 'conv4_2' + suffix, conv)
    setattr(n, 'relu4_2' + suffix, relu)
    
    conv, relu = conv_relu(relu, 512, pad=pad, param_name='conv4_3')
    setattr(n, 'conv4_3' + suffix, conv)
    setattr(n, 'relu4_3' + suffix, relu)


    if use_dilated_net:
	dilation = 2
	pool = relu
    else:
	dilation = 1
	pool = max_pool(relu)
	setattr(n, 'pool4' + suffix, pool)    
    
    conv, relu = conv_relu(pool, 512, pad=pad*dilation, dilation=dilation, param_name='conv5_1')
    setattr(n, 'conv5_1' + suffix, conv)
    setattr(n, 'relu5_1' + suffix, relu)
    
    conv, relu = conv_relu(relu, 512, pad=pad*dilation, dilation=dilation, param_name='conv5_2')
    setattr(n, 'conv5_2' + suffix, conv)
    setattr(n, 'relu5_2' + suffix, relu)
    
    conv, relu = conv_relu(relu, 512, pad=pad*dilation+last_layer_pad, dilation=dilation, param_name='conv5_3')
    setattr(n, 'conv5_3' + suffix, conv)
    setattr(n, 'relu5_3' + suffix, relu)
    
    
    if add_pool5:
	pool = max_pool(relu)
	setattr(n, 'pool5' + suffix, pool)
    
    return n

def fcn8_net(n, split, cur_shape = None, next_shape = None, fc6_size = 4096, fc7_size = 4096, crop_params = None, initialize_fcs=False):     
    return_crop_params = True if crop_params is None else False
    #Create network
    if cur_shape is None or next_shape is None:
	concat_pad = np.zeros((2,))
    else:
      concat_pad = (next_shape - cur_shape)/2.0/16.0

    if not all(concat_pad == np.round(concat_pad)):
	raise Exception

    conv_vgg(n, n.next_im, suffix='n', last_layer_pad=0, first_layer_pad=100, use_dilated_net=False, add_pool5=True)
    
    # concatination
    if crop_params is not None:
        conv_vgg(n, n.masked_im, suffix='m', last_layer_pad=concat_pad, first_layer_pad=100, use_dilated_net=False, add_pool5=True)
        n.silence_cur = L.Silence(n.cur_im, ntop=0)
	n.concat1 = L.Concat(n.pool5m, n.pool5n)
	vgg_out = n.concat1
    else:
	vgg_out = n.pool5n
    
     # fully conv
    if initialize_fcs:
	std = .01
    else:
	std = -1
	
    n.fc6, n.relu6 = conv_relu(vgg_out, fc6_size, ks=7, std=std)  
    if split == 'train':
        n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
        n.fc7, n.relu7 = conv_relu(n.drop6, fc7_size, ks=1, std=std)
        n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)    
    else:
        n.fc7, n.relu7 = conv_relu(n.relu6, fc7_size, ks=1)
    
    if initialize_fcs:
	n.score_fc = L.Convolution(n.relu7, kernel_size=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='gaussian', std=std), num_output=21)
    else:
	n.score_fc = L.Convolution(n.relu7, kernel_size=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], num_output=21)
	
    n.upscore2 = L.Deconvolution(n.score_fc, convolution_param=dict(kernel_size=4, stride=2, num_output=21), param=dict(lr_mult=0))
    n.score_pool4 = L.Convolution(n.pool4n, kernel_size=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], num_output=21)
    if return_crop_params:
	crop_params = [None] * 3
	ax, a, b = coord_map_from_to(n.score_pool4, n.upscore2)
	assert (a == 1).all(), 'scale mismatch on crop (a = {})'.format(a)
	assert (b <= 0).all(), 'cannot crop negative offset (b = {})'.format(b)
	assert (np.round(b) == b).all(), 'cannot crop noninteger offset (b = {})'.format(b)
	crop_params[0] = dict(axis=ax + 1,  # +1 for first cropping dim.
                                  offset=list(-np.round(b).astype(int)))
    
    n.score_pool4c = L.Crop(n.score_pool4, n.upscore2, crop_param=crop_params[0])
    n.fuse_pool4 = L.Eltwise(n.upscore2, n.score_pool4c, eltwise_param=dict(operation=P.Eltwise.SUM))
    
    
    
    n.upscore_pool4 = L.Deconvolution(n.fuse_pool4, convolution_param=dict(kernel_size=4, stride=2, num_output=21, bias_term=False), param=dict(lr_mult=0))
    n.score_pool3 = L.Convolution(n.pool3n, kernel_size=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], num_output=21)
    
    if return_crop_params:
	ax, a, b = coord_map_from_to(n.score_pool3, n.upscore_pool4)
	assert (a == 1).all(), 'scale mismatch on crop (a = {})'.format(a)
	assert (b <= 0).all(), 'cannot crop negative offset (b = {})'.format(b)
	assert (np.round(b) == b).all(), 'cannot crop noninteger offset (b = {})'.format(b)
	crop_params[1] = dict(axis=ax + 1,  # +1 for first cropping dim.
                                  offset=list(-np.round(b).astype(int)))
    
    n.score_pool3c = L.Crop(n.score_pool3, n.upscore_pool4, crop_param=crop_params[1])
    n.fuse_pool3 = L.Eltwise(n.upscore_pool4, n.score_pool3c, eltwise_param=dict(operation=P.Eltwise.SUM))
    
    
    n.upscore8 = L.Deconvolution(n.fuse_pool3, convolution_param=dict(kernel_size=16, stride=8, num_output=21, bias_term=False), param=dict(lr_mult=0))
    if return_crop_params:
	ax, a, b = coord_map_from_to(n.upscore8, n.next_im)
	assert (a == 1).all(), 'scale mismatch on crop (a = {})'.format(a)
	assert (b <= 0).all(), 'cannot crop negative offset (b = {})'.format(b)
	assert (np.round(b) == b).all(), 'cannot crop noninteger offset (b = {})'.format(b)
	crop_params[2] = dict(axis=ax + 1,  # +1 for first cropping dim.
                                  offset=list(-np.round(b).astype(int)))
    
    n.score_pool3c = L.Crop(n.upscore8, n.next_im, crop_param=crop_params[2])
    
    if initialize_fcs:
	n.score = L.Convolution(n.score_pool3c, kernel_size=1, param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)], weight_filler=dict(type='gaussian', std=std), num_output=2)
    else:
	n.score = L.Convolution(n.score_pool3c, kernel_size=1, param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)], num_output=2)
    return crop_params if return_crop_params else n


def simple_net(n, split, fc6_size = 4096, fc7_size = 4096, initialize_fcs=False, cur_shape = None, next_shape = None):
  
    #Get crop layer parameters
    tmp_net = caffe.NetSpec()
    tmp_net.im, tmp_net.label = L.MemoryData(batch_size=1, channels=3, height=244, width=244, ntop=2)
    conv_vgg(tmp_net, tmp_net.im, suffix='', last_layer_pad=0, first_layer_pad=100)
    tmp_net.fc6, tmp_net.relu6 = conv_relu(tmp_net.conv5_3, 4096, ks=7, dilation=4)        
    tmp_net.fc7, tmp_net.relu7 = conv_relu(tmp_net.relu6, 4096, ks=1, pad=0)
    tmp_net.fc8 = L.Convolution(tmp_net.relu7, kernel_size=1, num_output=2)
    tmp_net.upscore = L.Deconvolution(tmp_net.fc8, convolution_param=dict(kernel_size=16, stride=8, num_output=2))
    
    ax, a, b = coord_map_from_to(tmp_net.upscore, tmp_net.im)
    assert (a == 1).all(), 'scale mismatch on crop (a = {})'.format(a)
    assert (b <= 0).all(), 'cannot crop negative offset (b = {})'.format(b)
    assert (np.round(b) == b).all(), 'cannot crop noninteger offset (b = {})'.format(b)
    #
    
    #Create network
    if cur_shape is None or next_shape is None:
	concat_pad = np.zeros((2,))
    else:
      concat_pad = (next_shape - cur_shape)/2.0/8.0
    if not all(concat_pad == np.round(concat_pad)):
	raise Exception

    

    conv_vgg(n, n.cur_im, suffix='c', last_layer_pad=concat_pad, first_layer_pad=100)
    conv_vgg(n, n.masked_im, suffix='m', last_layer_pad=concat_pad, first_layer_pad=100)
    conv_vgg(n, n.next_im, suffix='n', last_layer_pad=0, first_layer_pad=100)
    
    # concatination
    n.concat1 = L.Concat(n.relu5_3c, n.relu5_3m, n.relu5_3n)
    
    # fully conv
    if initialize_fcs:
	std = .01
    else:
	std = -1
    n.fc6, n.relu6 = conv_relu(n.concat1, fc6_size, ks=7, dilation=4, std=std)
    if split == 'train':
        n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
        n.fc7, n.relu7 = conv_relu(n.drop6, fc7_size, ks=1, pad=0, std=std)
        n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
        n.fc8 = L.Convolution(n.drop7, kernel_size=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], num_output=2)
    else:
        n.fc7, n.relu7 = conv_relu(n.relu6, fc7_size, ks=1, pad=0, std=std)
        if initialize_fcs:
            n.fc8 = L.Convolution(n.relu7, kernel_size=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='gaussian', std=std), num_output=2)
        else:
            n.fc8 = L.Convolution(n.relu7, kernel_size=1, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], num_output=2)
        
    
    n.upscore = L.Deconvolution(n.fc8, convolution_param=dict(kernel_size=16, stride=8, num_output=2, group=2, weight_filler=dict(type='bilinear'),
                                                              bias_term=False), param=dict(lr_mult=0, decay_mult=0))

    n.score = L.Crop(n.upscore, n.next_im,
                  crop_param=dict(axis=ax + 1,  # +1 for first cropping dim.
                                  offset=list(-np.round(b).astype(int))))


def deepmask_net(n, split, cur_shape = None, next_shape = None, fc6_size = 512, fc7_size = 512, fc6_kernel = 3, upscore_scale = 4, initialize_fcs = False):
  
    
    #Create network
    if cur_shape is None or next_shape is None:
	concat_pad = np.zeros((2,))
    else:
      concat_pad = (next_shape - cur_shape)/2.0/16.0
    if not all(concat_pad == np.round(concat_pad)):
	print concat_pad
	raise Exception

    
    conv_vgg(n, n.cur_im, suffix='c', last_layer_pad=concat_pad, use_dilated_net = False)
    conv_vgg(n, n.masked_im, suffix='m', last_layer_pad=concat_pad, use_dilated_net = False)
    conv_vgg(n, n.next_im, suffix='n', last_layer_pad=0, use_dilated_net = False)
    
    # concatination
    n.concat1 = L.Concat(n.relu5_3c, n.relu5_3m, n.relu5_3n)
    
    inner_product_param = dict()
    if initialize_fcs:
	std = .01
	inner_product_param['weight_filler'] = dict(type='gaussian', std=std)
    else:
	std = -1
    n.fc6, n.relu6 = conv_relu(n.concat1, fc6_size, pad = (fc6_kernel - 1)/2, ks=fc6_kernel, std=std)
    n.fc7 = L.InnerProduct(n.fc6, num_output=fc7_size, **inner_product_param)
    n.fc8 = L.InnerProduct(n.fc7, num_output=2*next_shape[0]*next_shape[1]/upscore_scale/upscore_scale, **inner_product_param)
    
    
    #reshape Nx(2xscore_hxscore_w) to Nx2xscore_wxscore_h
    n.reshape = L.Reshape(n.fc8, reshape_param=dict(shape=dict(dim=[0, 2, next_shape[0]/upscore_scale, next_shape[1]/upscore_scale])))
    
    n.score = L.Deconvolution(n.reshape, convolution_param=dict(kernel_size=2 * upscore_scale - upscore_scale % 2, stride=upscore_scale, pad=int(np.ceil((upscore_scale - 1)/2.0)), num_output=2, group=2, weight_filler=dict(type='bilinear'),
                                                              bias_term=False), param=dict(lr_mult=0, decay_mult=0))

    
def create_net(split, net_info, datalayer_info):
    
    n = caffe.NetSpec()
    
    #Create datalayer
    if split != 'deploy':
	if datalayer_info['name'] == 'coco':
	    if np.any(datalayer_info['cur_shape'] * 2 != datalayer_info['next_shape']):
		raise Exception('Can not resize images into the given sizes')
	    
	    if split == 'train':
		coco_split = 'training' #Because Amir Is a Stupid Person!
	    elif split == 'val':
		coco_split = 'test'
	
	    pydata_params = dict(batch_size=datalayer_info['batch_size'], 
				im_shape=tuple(datalayer_info['next_shape']), 
				num_threads=datalayer_info['num_threads'], 
				max_queue_size=datalayer_info['max_queue_size'], split=coco_split, mean=tuple(datalayer_info['mean']),
				data_dir=datalayer_info['data_dir'],
				noisy_mask=datalayer_info['noisy_mask'])
	    module = 'coco_transformed_datalayers_prefetch'
	    layer = 'CocoTransformedDataLayerPrefetch'

	    n.cur_im, n.masked_im, n.next_im, n.label = L.Python(module=module, layer=layer, ntop=4, param_str=str(pydata_params))
	elif datalayer_info['name'] == 'davis':
	    if split == 'train':
		davis_split = 'training' #Because Amir Is a Stupid Person!
	    elif split == 'val':
		davis_split = 'test'
	    pydata_params = dict(batch_size=datalayer_info['batch_size'], 
				im_shape=tuple(datalayer_info['next_shape']), 
				num_threads=datalayer_info['num_threads'], 
				max_queue_size=datalayer_info['max_queue_size'], split=davis_split, mean=tuple(datalayer_info['mean']))
	    module = 'davis_datalayer_prefetch_randomer'
	    layer = 'DavisDataLayerPrefetch'
	    
	    n.cur_im, n.masked_im, n.next_im, n.label = L.Python(module=module, layer=layer, ntop=4, param_str=str(pydata_params))
	    
	elif datalayer_info['name'] == 'davis_sequence':
	    if split == 'train':
		davis_split = 'training' #Because Amir Is a Stupid Guy!
	    elif split == 'val':
		davis_split = 'test'
	    pydata_params = dict(batch_size=datalayer_info['batch_size'], 
				im_shape=tuple(datalayer_info['next_shape']), 
				split=davis_split,
				port = datalayer_info['port'],
				mean=tuple(datalayer_info['mean']))
	    if datalayer_info.has_key('shuffle'):
		pydata_params['shuffle'] = datalayer_info['shuffle']
	    if datalayer_info.has_key('max_len'):
		pydata_params['max_len'] = datalayer_info['max_len']
	    module = 'davis_datalayer_server'
	    layer = 'DavisDataLayerServer'
	    
	    n.cur_im, n.masked_im, n.next_im, n.label = L.Python(module=module, layer=layer, ntop=4, param_str=str(pydata_params))
	else:
	    raise Exception
    elif split == 'deploy':
         n.cur_im, n.label_1 = L.MemoryData(batch_size=datalayer_info['batch_size'], channels=3, height=datalayer_info['cur_shape'][0], width=datalayer_info['cur_shape'][1], ntop=2)
         n.masked_im, n.label_2 = L.MemoryData(batch_size=datalayer_info['batch_size'], channels=3, height=datalayer_info['cur_shape'][0], width=datalayer_info['cur_shape'][1], ntop=2)
         n.next_im, n.label_3 = L.MemoryData(batch_size=datalayer_info['batch_size'], channels=3, height=datalayer_info['next_shape'][0], width=datalayer_info['next_shape'][1], ntop=2)
    else:
        raise Exception

    #Create net
    if net_info['name'] == 'simple':
	simple_net(n, split, fc6_size = net_info['fc6_size'], fc7_size = net_info['fc7_size'], 
	    initialize_fcs=net_info['initialize_fcs'], cur_shape = datalayer_info['cur_shape'], 
	    next_shape = datalayer_info['next_shape'])    
    elif net_info['name'] == 'deepmask':
	deepmask_net(n, split, cur_shape = datalayer_info['cur_shape'], next_shape = datalayer_info['next_shape'], 
	      fc6_size = net_info['fc6_size'], fc7_size = net_info['fc7_size'], initialize_fcs=net_info['initialize_fcs'], 
	      fc6_kernel = net_info['fc6_kernel'], upscore_scale = net_info['upscore_scale'])
    elif net_info['name'] == 'fcn8':
	tmp_net = caffe.NetSpec()
	tmp_net.next_im, tmp_net.label = L.MemoryData(batch_size=1, channels=3, height=244, width=244, ntop=2)
	crop_params = fcn8_net(tmp_net, split, cur_shape = datalayer_info['cur_shape'], next_shape = datalayer_info['next_shape'], 
		fc6_size = net_info['fc6_size'], fc7_size = net_info['fc7_size'], crop_params = None)
	fcn8_net(n, split, cur_shape = datalayer_info['cur_shape'], next_shape = datalayer_info['next_shape'], 
		fc6_size = net_info['fc6_size'], fc7_size = net_info['fc7_size'], crop_params = crop_params, initialize_fcs=net_info['initialize_fcs'])
    
    mask_blob = n.score
    if net_info.has_key('enhancement_method'):
	if net_info['enhancement_method'] == 'crf':
	    n.crf_score = L.MultiStageMeanfield(mask_blob, mask_blob, n.next_im, param=[dict(lr_mult=0.001, decay_mult=1), 
						dict(lr_mult=0.001, decay_mult=1), dict(lr_mult=0.01, decay_mult=1)], 
						multi_stage_meanfield_param=dict(num_iterations=10,compatibility_mode=0,threshold=2,
						theta_alpha=160,theta_beta=3.0/255,theta_gamma=3,spatial_filter_weight=3,
						bilateral_filter_weight=5))
	    mask_blob = n.crf_score
	
    if split != 'deploy':
	n.loss = L.SoftmaxWithLoss(mask_blob, n.label,
                                   loss_param=dict(ignore_label=255))
    else:
        n.prop = L.Softmax(mask_blob)
    
    if datalayer_info['name'] == 'davis_sequence':
	pydata_params = dict(port=datalayer_info['port'])
	module = 'davis_datalayer_client'
	layer = 'DavisDataLayerClient'
	n.clinet_top = L.Python(mask_blob, module=module, layer=layer, ntop=1, param_str=str(pydata_params))
    return n
