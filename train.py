import  torch, os
import  numpy as np
from    dataloaders.MiniImagenet import MiniImagenet
from    dataloaders.cifarfs_loader import CifarFS
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
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
import logging

def cycle(iterable):
	iterator = iter(iterable)
	while True:
		try:
			yield next(iterator)
		except StopIteration:
			iterator = iter(iterable)

def main(args):
	store_dir = os.path.dirname(args.restore_model) if args.restore_model else os.path.join(args.base_results_dir, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
	# store_dir = os.path.join(args.base_results_dir,time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
	os.makedirs(store_dir,exist_ok=True)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	# torch.backends.cudnn.deterministic = True
	np.random.seed(args.seed)
	random.seed(args.seed)

	log_format = '%(levelname)-8s %(message)s'

	log_file_name = 'train.log'

	logfile = os.path.join(store_dir, log_file_name)
	logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
	logging.getLogger().addHandler(logging.StreamHandler())
	logging.info(json.dumps(args.__dict__))

	writer = SummaryWriter(log_dir=store_dir+'/'+args.comment)

	with open(os.path.join(store_dir,'args.json'),'w+') as args_file:
		json.dump(args.__dict__, args_file)

	if args.svdo:
		config = [
			('Conv2d_SVDO', [args.conv_layers, 3, 3, 3, 1, 1]),
			('bn', [args.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 2, 0]),
			('Conv2d_SVDO', [args.conv_layers, args.conv_layers, 3, 3, 1, 1]),
			('bn', [args.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 2, 0]),
			('Conv2d_SVDO', [args.conv_layers, args.conv_layers, 3, 3, 1, 1]),
			('bn', [args.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 2, 0]),
			('Conv2d_SVDO', [args.conv_layers, args.conv_layers, 3, 3, 1, 1]),
			('bn', [args.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 1, 0]),
			('flatten', []),
			('Linear_SVDO', [args.n_way, args.conv_layers * 3 * 3])
		]
	else:
		config = [
			('Conv2d', [args.conv_layers, 3, 3, 3, 1, 1]),
			('bn', [args.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 2, 0]),
			('Conv2d', [args.conv_layers, args.conv_layers, 3, 3, 1, 1]),
			('bn', [args.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 2, 0]),
			('Conv2d', [args.conv_layers, args.conv_layers, 3, 3, 1, 1]),
			('bn', [args.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 2, 0]),
			('Conv2d', [args.conv_layers, args.conv_layers, 3, 3, 1, 1]),
			('bn', [args.conv_layers]),
			('relu', [True]),
			('MaxPool2d', [2, 1, 0]),
			('flatten', []),
			('Linear', [args.n_way, args.conv_layers * 3 * 3])
		]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logging.info(device)

	if args.pretrained:
		assert(args.custom==False)


	model = MyModel(config,threshold=args.threshold,init_linsvdo = args.init_linsvdo,init_convsvdo = args.init_convsvdo, pretrained=args.pretrained,svdo=args.svdo,custom=True)#args.custom)
	model.train()
	model = model.to(device)

	if args.restore_model:
		logging.info('Resuming Training..........................')
		model_checkpoint = torch.load(args.restore_model, map_location=device)
		model.load_state_dict(model_checkpoint['state_dict'])

		args.kl_weight = model_checkpoint['kl_weight']
		args.start_step = model_checkpoint['step']
		args.start_test_step = model_checkpoint['test_step']
		logging.info('start_step:{},start_test:{}'.format(args.start_step,args.start_test_step))
		# args.

	logging.info(model.model_layers)
	# add other datasets

	start_time = time.time()
	if args.dset=='mini':
		data_train = MiniImagenet(args.dset_path, phase='train', n_way=args.n_way, k_spt=args.k_spt, k_query=args.k_qry, imgsz=args.imgsz)
		data_test = MiniImagenet(args.dset_path, phase='val', n_way=args.n_way, k_spt=args.k_spt, k_query=args.k_qry, imgsz=args.imgsz)
	elif args.dset=='cifarfs':
		data_train = CifarFS(args.dset_path, phase='train', n_way=args.n_way, k_spt=args.k_spt, k_query=args.k_qry, imgsz=args.imgsz)
		data_test = CifarFS(args.dset_path, phase='val', n_way=args.n_way, k_spt=args.k_spt, k_query=args.k_qry, imgsz=args.imgsz)

	db_train = cycle(DataLoader(data_train, batch_size=args.meta_batchsz, shuffle=True, pin_memory=True))
	db_test = DataLoader(data_test, batch_size=1, shuffle=False, pin_memory=True)
	logging.info('len(db_test);{}'.format(len(db_test)))

	end_time = time.time()
	logging.info('Time to load data:{}'.format(end_time-start_time))
	# x_spt, y_spt, x_qry, y_qry  = next(iter(db))
	if args.train:
		tmp = filter(lambda x: x.requires_grad, model.parameters())    # contains all parameters which require gradient among all network params
		num = sum(map(lambda x: np.prod(x.shape), tmp)) # sum of shapes of each x in tmp params -> gives total number of gradient requiring(trainable) params
		# print()
		logging.info('Total trainable tensors:{}'.format(num))
		# for step in range(num_iterations):
		#     x_spt,y_spt,x_qry,y_qry = next(db_train)
		meta_optim = optim.Adam(model.parameters(), lr=args.meta_lr)

		if args.restore_model:
			kl_weight = args.kl_weight
			optim_file_name = '/optim_checkpoint_step'+str(args.start_step)+'.pt'
			meta_optim.load_state_dict(torch.load(os.path.dirname(args.restore_model) + optim_file_name, map_location=device))
			logging.info('Loaded OPtimizer state!')
		else:
			kl_weight = min(args.initial_kl_weight+1e-6, 0.01)
		global_test_steps=args.start_test_step

		for step in range(args.start_step,args.num_iterations):
			x_spt, y_spt, x_qry, y_qry = next(iter(db_train))
			x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

			accs,tr_loss = meta_grad_comp(model,args, x_spt, y_spt, x_qry, y_qry,kl_weight,meta_optim,phase='train')
			if args.svdo:
				sp_layer_names,sp_vals,dropout_vals = model.get_model_sparsity()
				for idx,layer_name in enumerate(sp_layer_names):
					writer.add_scalar('Sparsity_'+layer_name,sp_vals[idx],step)
				
			writer.add_scalar('Train_Loss',tr_loss[-1],step)
			writer.add_scalar('Train_Acc',accs[-1],step)
			if step % 1000==0:
				kl_weight = min(kl_weight+1e-6, 1e-5)


			if step%20==0:
				logging.info('Step:{}, Training Acc:{}, Training Loss:{}'.format(step,accs,tr_loss))
				if args.svdo:
					logging.info('Sparsity:{}'.format(sp_vals))

			if step%500==0:
				torch.save({'kl_weight':kl_weight,
						'step': step,
						'test_step':global_test_steps,
						# 'optim':meta_optim,
						'state_dict': model.state_dict()},
						os.path.join(store_dir, 'step_'+str(step)+'_model_checkpoint.pt'))
				torch.save(meta_optim.state_dict(), os.path.join(store_dir, 'optim_checkpoint_step'+str(step)+'.pt'))


			if  step%500==0:
				torch.manual_seed(args.val_seed)
				torch.cuda.manual_seed_all(args.val_seed)
				# torch.backends.cudnn.deterministic = True
				np.random.seed(args.val_seed)
				random.seed(args.val_seed)
				step_test= 0 

				net = MyModel(config,threshold=args.threshold,init_linsvdo = args.init_linsvdo,init_convsvdo = args.init_convsvdo, pretrained=args.pretrained,svdo=args.svdo,custom=True)
				net.load_state_dict(deepcopy(model.state_dict()))
				net = net.to(device)


				accs_all_test = []
				loss_all_test = []
				st = time.time()
				while(step_test<args.max_steps_test):
					print('Running Test Step:{}/{}'.format(step_test+1,args.max_steps_test),end='\r')
					x_spt, y_spt, x_qry, y_qry = next(iter(db_test))

					x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
												 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

					accs,te_loss = maml_finetune(net,args, x_spt, y_spt, x_qry, y_qry,kl_weight,phase='test')

					accs_all_test.append(accs)
					loss_all_test.append(te_loss)
					step_test+=1


				accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
				test_loss = np.array(loss_all_test).mean(axis=0).astype(np.float16)
				logging.info('Test Loss:{}'.format(test_loss))
				writer.add_scalar('Test_Loss',test_loss[-1],global_test_steps)
				writer.add_scalar('Test_Acc',accs[-1],global_test_steps)                
				logging.info('Test acc:{}'.format(accs))
				del net
				global_test_steps+=1
				end=time.time()
				logging.info('Time taken for test:{}'.format(end-st))

				torch.manual_seed(args.seed)
				torch.cuda.manual_seed_all(args.seed)
				# torch.backends.cudnn.deterministic = True
				np.random.seed(args.seed)
				random.seed(args.seed)
	


if __name__ == '__main__':

	argparser = argparse.ArgumentParser()
	argparser.add_argument('--base_results_dir',help='Enter path for base directory to store results', default='/media/tiwari/My Passport/lokender/anish/maml_sparse/results/cifar_fs_5_way_1_shot/')
	argparser.add_argument('--dset_path',default='/media/tiwari/My Passport/lokender/anish/maml_sparse/data/CIFAR-FS/')
	argparser.add_argument('--dset',default='mini', help='Choose from `mini` (MiniImagenet) or `cifarfs` ')
	argparser.add_argument('--restore_model',type=str, help='Path to model to restore.',default=None)

	argparser.add_argument('--num_iterations', type=int, help='epoch number', default=60000)
	argparser.add_argument('--conv_layers',type=int,default=32)
	argparser.add_argument('--seed',type=int,default=100)
	argparser.add_argument('--val_seed',type=int,default=5)


	argparser.add_argument('--n_way', type=int, help='n way', default=5)
	argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
	argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
	argparser.add_argument('--imgsz', type=int, help='imgsz', default=32)
	argparser.add_argument('--imgc', type=int, help='imgc', default=3)
	argparser.add_argument('--meta_batchsz', type=int, help='meta batch size, namely task num', default=4)
	argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
	argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2)
	argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
	argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
	argparser.add_argument('--threshold', type=float, help='for removing weights (SVD)', default=3.0)
	argparser.add_argument('--initial_kl_weight',type=float,default=1e-6, help='Coeff for KL Loss')
	argparser.add_argument('--max_steps_test',type=int,default=600)
	argparser.add_argument('--start_step',type=int,default=0)
	argparser.add_argument('--start_test_step',type=int,default=0)

	argparser.add_argument('--init_linsvdo',type=float,default=-8.0,help='Param init value for linearSVDO')
	argparser.add_argument('--init_convsvdo',type=float,default=-8.0,help='Param init value for convSVDO')

	argparser.add_argument('--pretrained',action='store_true')
	argparser.add_argument('--svdo',action='store_true',help='Invoking this param implies using Sparse VD on MAML')
	argparser.add_argument('--custom',action='store_false',help='Switch off if you want to use models like vgg11_bn')    # custom=False doesn't work yet


	argparser.add_argument('--train', action='store_true', default=True)
	argparser.add_argument('--resume',action='store_true',default=False)



	args = argparser.parse_args()

	pre_suffix,svdo_suffix = '',''
	if args.pretrained:
		pre_suffix = 'pretrained_'
	if args.svdo:
		svdo_suffix = 'svdo'

	if not args.pretrained and not args.svdo:
		args.comment = str(args.n_way)+'_way_'+str(args.k_spt)+'_shot_'+args.dset_path.split('/')[-2]
	else:
		args.comment = str(args.n_way)+'_way_'+str(args.k_spt)+'_shot_'+args.dset_path.split('/')[-2]+'_'+pre_suffix+svdo_suffix
	main(args)
