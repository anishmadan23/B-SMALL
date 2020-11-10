import  torch, os
import  numpy as np
from    dataloaders.MiniImagenet import MiniImagenet
from    dataloaders.cifarfs_loader import CifarFS
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
import logging
import time
import json
from tensorboardX import SummaryWriter
from utils import *
import itertools as it 
import time
from torch import optim
import random
from models.myModel import MyModel
from grad_comp import meta_grad_comp,maml_finetune  
from copy import deepcopy
from utils import * 

def mean_confidence_interval(accs, confidence=0.95):
	n = accs.shape[0]
	stds = np.std(accs, 0)
	ci95 = 1.96*stds/np.sqrt(n)
	return ci95

def main(params,args):
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	log_format = '%(levelname)-8s %(message)s'

	log_file_name = 'test.log'

	logfile = os.path.join(os.path.dirname(args.restore_model), log_file_name)
	logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
	logging.getLogger().addHandler(logging.StreamHandler())
	logging.info(json.dumps(params.__dict__))

	logging.info('New Test Args \n')
	logging.info(json.dumps(args.__dict__))

	if params.svdo:
		config = [
			('Conv2d_SVDO', [params.conv_layers, 3, 3, 3, 1, 1]),
			('bn', [params.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 2, 0]),
			('Conv2d_SVDO', [params.conv_layers, params.conv_layers, 3, 3, 1, 1]),
			('bn', [params.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 2, 0]),
			('Conv2d_SVDO', [params.conv_layers, params.conv_layers, 3, 3, 1, 1]),
			('bn', [params.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 2, 0]),
			('Conv2d_SVDO', [params.conv_layers, params.conv_layers, 3, 3, 1, 1]),
			('bn', [params.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 1, 0]),
			('flatten', []),
			('Linear_SVDO', [params.n_way, params.conv_layers * 3 * 3])
		]
	else:
		config = [
			('Conv2d', [params.conv_layers, 3, 3, 3, 1, 1]),
			('bn', [params.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 2, 0]),
			('Conv2d', [params.conv_layers, params.conv_layers, 3, 3, 1, 1]),
			('bn', [params.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 2, 0]),
			('Conv2d', [params.conv_layers, params.conv_layers, 3, 3, 1, 1]),
			('bn', [params.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 2, 0]),
			('Conv2d', [params.conv_layers, params.conv_layers, 3, 3, 1, 1]),
			('bn', [params.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 1, 0]),
			('flatten', []),
			('Linear', [params.n_way, params.conv_layers * 3 * 3])
		]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logging.info(device)
	if params.pretrained:
		assert(params.custom==False)

	model = MyModel(config,threshold=params.threshold,init_linsvdo = params.init_linsvdo,init_convsvdo = params.init_convsvdo, pretrained=params.pretrained,svdo=params.svdo,custom=True)#args.custom)
	model = model.to(device)

	logging.info('Loading Model..........................')
	model_checkpoint = torch.load(args.restore_model, map_location=device)
	model.load_state_dict(model_checkpoint['state_dict'])
	args.kl_weight = model_checkpoint['kl_weight']

	if params.dset=='cifarfs':
		data_test = CifarFS(params.dset_path, phase='test', n_way=params.n_way, k_spt=params.k_spt, k_query=params.k_qry, imgsz=params.imgsz)
	else:
		data_test = MiniImagenet(params.dset_path, phase='test', n_way=params.n_way, k_spt=params.k_spt, k_query=params.k_qry, imgsz=params.imgsz)
	db_test = DataLoader(data_test, batch_size=1, shuffle=False, pin_memory=True)
	logging.info('len(db_test);{}'.format(len(db_test)))

	
	### check model sparsity ####
	
	step_test = 0
	if args.max_steps_test==-1:
		args.max_steps_test = len(db_test)
	print(args.max_steps_test)

	accs_all_test = []
	loss_all_test = []
	avg_sparsity = []
	while(step_test<args.max_steps_test):
		print('Running Test Step:{}/{}'.format(step_test+1,args.max_steps_test),end='\r')
		x_spt, y_spt, x_qry, y_qry = next(iter(db_test))

			# print('Test',x_spt.shape,y_sptparams.dset_path.shape,x_qry.shape,y_qry.shape)
		x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
									 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

		accs,te_loss = maml_finetune(model,params, x_spt, y_spt, x_qry, y_qry,args.kl_weight,phase='test')
		if params.svdo:
			sp_layer_names,sp_vals,dropout_vals = model.get_model_sparsity()

			all_w = sum([x.size for x in list(dropout_vals.values())])
			dropped_out_w = sum([(x>params.threshold).sum() for x in list(dropout_vals.values())])
			sparsity = dropped_out_w/all_w *100
			avg_sparsity.append(sparsity)

		if step_test%50==0:
			logging.info('Step:{}, Accs:{}'.format(step_test,accs))

		accs_all_test.append(np.array(accs))
		loss_all_test.append(te_loss)
		step_test+=1


	accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
	test_loss = np.array(loss_all_test).mean(axis=0).astype(np.float16)
	logging.info('Final Test Loss:{}'.format(test_loss))               
	logging.info('Final Test acc:{}'.format(accs))

	ci95 = mean_confidence_interval(np.array(accs_all_test))
	logging.info('Confidence Interval Estimate')
	logging.info(ci95*100)

	if params.svdo:
		logging.info('Model Sparsity:{}'.format(np.mean(avg_sparsity)))


if __name__=='__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--base_results_dir',default='/media/tiwari/My Passport/lokender/anish/maml_sparse/results/')
	argparser.add_argument('--dset_path',default='/media/tiwari/My Passport/lokender/anish/maml_sparse/dataloaders/CIFAR-FS/')
	argparser.add_argument('--restore_model',type=str, help='Path to model to restore.',required=True)
	argparser.add_argument('--max_steps_test',type=int,default=600)
	argparser.add_argument('--seed',type=int,default=1,help='Seed for test set')
	# argparser.add_argument('--conv_layers',type=int,default=32)

	args = argparser.parse_args()

	params = Dict2Obj(json.load(
			open(os.path.join(os.path.dirname(args.restore_model), "args.json"), "r")))
	main(params,args)