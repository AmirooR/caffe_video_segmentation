import caffe

from caffe import layers as L, params as P
from caffe.coord_map import coord_map_from_to
from surgery import *
import numpy as np
from net import create_net, save_net
import string
import os, errno

CAFFE_PATH = '/home/amir/caffe/build/tools/caffe'
COCO_PATH = '/home/amir/coco'
def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occured

	
def fcn8_solver_param(datalayer_name):
    solver_param = dict(momentum=.9,iter_size=1,weight_decay=5e-4, snapshot_iter=5e3, 
			weights_path='./fcn8_net.caffemodel',
			caffe_path=CAFFE_PATH, 
			resume_state_path='')
    if datalayer_name == 'coco':
	solver_param['max_iter'] = 700e3
	solver_param['lr_policy_step_size'] = 100e3
	solver_param['snapshot_iter'] = 1e4
	solver_param['base_lr'] = 1e-5
    elif datalayer_name[:5] == 'davis':
	solver_param['iter_size']=1
	solver_param['max_iter'] = 1e5
	solver_param['lr_policy_step_size'] = 10e3
	solver_param['base_lr'] = 1e-5
    return solver_param

def simple_solver_param(datalayer_name):
    solver_param = dict(momentum=.9,iter_size=5,weight_decay=5e-4, snapshot_iter=5e3, 
			weights_path='./simple_net.caffemodel',
			caffe_path=CAFFE_PATH, 
			resume_state_path='')
    if datalayer_name == 'coco':
	solver_param['max_iter'] = 350e3
	solver_param['lr_policy_step_size'] = 100e3
	solver_param['base_lr'] = 1e-3
    elif datalayer_name[:5] == 'davis':
	solver_param['iter_size']=1
	solver_param['max_iter'] = 1e5
	solver_param['lr_policy_step_size'] = 10e3
	solver_param['base_lr'] = 1e-5
    return solver_param

def deepmask_solver_param(datalayer_name):
    solver_param = dict(momentum=.9,iter_size=1,weight_decay=5e-4, snapshot_iter=5e4, 
			weights_path='./deepmask_net.caffemodel',
			caffe_path=CAFFE_PATH, 
			resume_state_path='')
    if datalayer_name == 'coco':
	solver_param['max_iter'] = 1e6
	solver_param['lr_policy_step_size'] = 300e3
	solver_param['base_lr'] = 1e-3
    elif datalayer_name[:5] == 'davis':
	solver_param['max_iter'] = 10e3
	solver_param['lr_policy_step_size'] = 10e3
	solver_param['base_lr'] = 1e-5 
    return solver_param


def fcn8_net_info(initialize_fcs=False, datalayer_name = 'coco'):
    if datalayer_name == 'davis_sequence':
	datalayer_info = dict(name=datalayer_name, cur_shape = np.array([384,384])/ 2, next_shape = np.array([384,384]), batch_size=3, port = '6678', shuffle = True, max_len = 20, mean = (0.48501961, 0.45795686,0.40760392))
    else:
	datalayer_info = dict(name=datalayer_name, cur_shape = np.array([384,384])/ 2, next_shape = np.array([384,384]), batch_size=6,num_threads=2,max_queue_size = 8, mean = (0.48501961, 0.45795686,0.40760392), data_dir=COCO_PATH, noisy_mask = True)
    
    net_info = dict(name='fcn8',fc6_size=4096, fc7_size = 4096, initialize_fcs=initialize_fcs, surgeon=fcn8_net_surgery)  
    solver_param = fcn8_solver_param(datalayer_name)
    return (datalayer_info, net_info, solver_param)

def simple_net_info(initialize_fcs=False, datalayer_name = 'coco'):
    if datalayer_name == 'davis_sequence':
	datalayer_info = dict(name=datalayer_name, cur_shape = np.array([384,384])/ 2, next_shape = np.array([384,384]), batch_size=4, port = '6678', shuffle = True, max_len = 10, mean = (0.48501961, 0.45795686,0.40760392))
    else:
	datalayer_info = dict(name=datalayer_name, cur_shape = np.array([384,384])/ 2, next_shape = np.array([384,384]), batch_size=4,num_threads=2,max_queue_size = 8, mean = (0.48501961, 0.45795686,0.40760392), data_dir=COCO_PATH, noisy_mask = True)
    
    net_info = dict(name='simple',fc6_size=1024, fc7_size = 1024, initialize_fcs=initialize_fcs, surgeon=simple_net_surgery)  
    solver_param = simple_solver_param(datalayer_name)
    return (datalayer_info, net_info, solver_param)

def deepmask_net_info(initialize_fcs=False, datalayer_name = 'coco'):
    if datalayer_name == 'davis_sequence':
	datalayer_info = dict(name=datalayer_name, cur_shape = np.array([384,384])/ 2, next_shape = np.array([384,384]), batch_size=4, port = '6678', mean = (0.48501961, 0.45795686,0.40760392), noisy_mask = True)
    else:
	datalayer_info = dict(name=datalayer_name, cur_shape = np.array([256,256])/ 2, next_shape = np.array([256,256]), batch_size=24,num_threads=4,max_queue_size = 48, data_dir=COCO_PATH)
    net_info = dict(name='deepmask',fc6_size=1024, fc7_size = 1024, initialize_fcs=initialize_fcs, surgeon=deepmask_net_surgery, fc6_kernel=3, upscore_scale = 4)
    solver_param = deepmask_solver_param(datalayer_name)
    return (datalayer_info, net_info, solver_param)

def parse_file(input_path, output_path, dictionary):
    with open(input_path, 'r') as in_file:
	with open(output_path, 'w') as out_file:
	    data = string.Template(in_file.read())
	    out_file.write(data.substitute(**dictionary))


def create_files(datalayer_info, net_info, solver_param, create_caffemodel = False, clean = False):
    
    splits = ['train', 'val']
    
    suffix = datalayer_info['name'] + '_' + net_info['name']
   
    if create_caffemodel and not clean:
	#Do Surgery to get parameters
	net_info['surgeon'](datalayer_info, net_info)
    
    #create net
    for split in splits:
	file_name = split + '_' + suffix + '.prototxt'
	if clean:
	    silentremove(file_name)
	else:
	    net_spec = create_net(split, net_info, datalayer_info)
	    save_net(file_name, str(net_spec.to_proto()))
    
   
    
    solver_path = 'solver_' + suffix + '.prototxt'
    train_path = 'train_' + suffix + '.sh'
    resume_path = 'resume_' + suffix + '.sh'
    if clean:
	silentremove(solver_path)
	silentremove(train_path)
	silentremove(resume_path)
    else:
	#create solver$suffix.prototxt
	final_iter_size =  solver_param['iter_size'] * datalayer_info['batch_size']
	solver_dict = dict(TRAIN_NET= 'train_' + suffix + '.prototxt', 
		       BASE_LR = solver_param['base_lr'],
		       STEP_SIZE= int(solver_param['lr_policy_step_size'] / final_iter_size), 
		       MOMENTUM= solver_param['momentum'], 
		       ITER_SIZE= solver_param['iter_size'], 
		       MAX_ITER= int(solver_param['max_iter'] / final_iter_size), 
		       WEIGHT_DECAY= solver_param['weight_decay'], 
		       SNAPSHOT= int(solver_param['snapshot_iter'] / final_iter_size),
		       SNAPSHOT_PREFIX= './snapshots/' + suffix)
	
	#create train$suffix.sh
	train_dict = dict(CAFFE_PATH=solver_param['caffe_path'],SOLVER=solver_path,WEIGHTS=solver_param['weights_path'], LOG_NAME='train_' + suffix)
	
	#create resume`$suffix.sh
	resume_dict = dict(CAFFE_PATH=solver_param['caffe_path'], SOLVER=solver_path, SNAPSHOT=solver_param['resume_state_path'], LOG_NAME='resume_' + suffix)
	
	parse_file('./data/solver.prototxt', solver_path, solver_dict)
	parse_file('./data/train.sh', train_path, train_dict)
	parse_file('./data/resume.sh', resume_path, resume_dict)
if __name__ == '__main__':
    
    
    net_info_retriver = fcn8_net_info
    clean = False
    
    datalayer_info, net_info, solver_param = net_info_retriver(datalayer_name = 'coco')
    create_files(datalayer_info, net_info, solver_param, True, clean)
    
    datalayer_info, net_info, solver_param = net_info_retriver(datalayer_name = 'davis')
    create_files(datalayer_info, net_info, solver_param, False, clean)
    
    datalayer_info, net_info, solver_param = net_info_retriver(datalayer_name = 'davis_sequence')
    net_info['enhancement_method'] = 'crf'
    create_files(datalayer_info, net_info, solver_param, False, clean)
