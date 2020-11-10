import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from models.myModel import MyModel
from copy import deepcopy


def meta_grad_comp(model,args, x_spt, y_spt, x_qry, y_qry,kl_weight,meta_optim,phase='train'):

	order = 2
	create_graph = (order==2) & (phase=='train')
	task_num, setsz, c_, h, w = x_spt.size()

	querysz = x_qry.size(1)

	losses_q = [0 for _ in range(args.update_step + 1)]  # losses_q[i] is the loss on step i
	# kl_q = [0 for _ in range(args.update_step + 1)]  
	corrects = [0 for _ in range(args.update_step + 1)]

	# task_gradients = []
	# print("ENTERING........")
	for i in range(args.meta_batchsz):              #iterate over batch of tasks
 
		logits = model(x_spt[i], vars=None, bn_training=True)
		sgvlb_loss = F.cross_entropy(logits, y_spt[i]) + kl_weight * model.kl_reg            

		grad = torch.autograd.grad(sgvlb_loss,model.parameters(),create_graph=True,retain_graph=True)
		assert(len(grad)==len(model.parameters()))

		fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, model.parameters())))

		with torch.no_grad():

			logits_q= model(x_qry[i], model.parameters(), bn_training=True)

			sgvlb_loss_q = F.cross_entropy(logits_q, y_qry[i]) + kl_weight * model.kl_reg           

			losses_q[0] += sgvlb_loss_q

			pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
			correct = torch.eq(pred_q, y_qry[i]).sum().item()
			corrects[0] = corrects[0] + correct

			# this is the loss and accuracy after the first update
		with torch.no_grad():
			logits_q = model(x_qry[i], fast_weights, bn_training=True)
 
			sgvlb_loss_q = F.cross_entropy(logits_q, y_qry[i]) + kl_weight * model.kl_reg            
			losses_q[1] += sgvlb_loss_q

			pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
			correct = torch.eq(pred_q, y_qry[i]).sum().item()
			corrects[1] = corrects[1] + correct

		for k in range(1, args.update_step):
			logits = model(x_spt[i], fast_weights, bn_training=True)

			sgvlb_loss = F.cross_entropy(logits, y_spt[i]) + kl_weight * model.kl_reg            


			grad = torch.autograd.grad(sgvlb_loss, fast_weights,create_graph=create_graph,retain_graph=True,allow_unused=True)
			assert(len(grad)==len(model.parameters()))

			grad2,grad3 = grad[0].clone(),grad[1].clone()

			fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, fast_weights)))

			logits_q = model(x_qry[i], fast_weights, bn_training=True)

			sgvlb_loss_q = F.cross_entropy(logits_q, y_qry[i]) + kl_weight * model.kl_reg        

			losses_q[k + 1] += sgvlb_loss_q

			with torch.no_grad():
				pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
				correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
				corrects[k + 1] = corrects[k + 1] + correct

	loss_q = losses_q[-1] / args.meta_batchsz
	model.train()
	meta_optim.zero_grad()
	loss_q.backward()

	meta_optim.step()
	# print('Corrects Train',corrects,task_num*args.n_way)
	accs = np.array(corrects) / (querysz*task_num)

	my_losses = [my_loss.cpu().item() for my_loss in losses_q]
	# print('My Losses',my_losses)
	avg_loss_q = np.array(my_losses)/args.meta_batchsz

	# print('avg loss',avg_loss_q)
	return accs,avg_loss_q

def maml_finetune(net,args, x_spt, y_spt, x_qry, y_qry,kl_weight,phase='test'):

	corrects = [0 for _ in range(args.update_step_test + 1)]
	losses_q = [0 for _ in range(args.update_step_test + 1)]  # losses_q[i] is the loss on step i

	order=2
	create_graph = (order==2) & (phase=='train')
	
	querysz = x_qry.size(0)
	logits = net(x_spt)

	sgvlb_loss = F.cross_entropy(logits, y_spt) + kl_weight * net.kl_reg            
	grad = torch.autograd.grad(sgvlb_loss, net.parameters())
	fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, net.parameters())))

	with torch.no_grad():
			# [setsz, nway]
		logits_q = net(x_qry, net.parameters(), bn_training=True)

		sgvlb_loss_q = F.cross_entropy(logits_q, y_qry) + kl_weight * net.kl_reg

		pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
		losses_q[0] += sgvlb_loss_q
		# scalar
		correct = torch.eq(pred_q, y_qry).sum().item()
		corrects[0] = corrects[0] + correct

		# this is the loss and accuracy after the first update
	with torch.no_grad():
		# [setsz, nway]
		logits_q = net(x_qry, fast_weights, bn_training=True)

		sgvlb_loss_q = F.cross_entropy(logits_q, y_qry) + kl_weight * net.kl_reg
		pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)

		losses_q[1] += sgvlb_loss_q
		# scalar
		correct = torch.eq(pred_q, y_qry).sum().item()
		corrects[1] = corrects[1] + correct


	for k in range(1, args.update_step_test):

		logits = net(x_spt, fast_weights, bn_training=True)

		sgvlb_loss = F.cross_entropy(logits, y_spt) + kl_weight * net.kl_reg           

		# 2. compute grad on theta_pi
		grad = torch.autograd.grad(sgvlb_loss, fast_weights)
		# 3. theta_pi = theta_pi - train_lr * grad
		fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(grad, fast_weights)))

		logits_q = net(x_qry, fast_weights, bn_training=True)

		sgvlb_loss_q = F.cross_entropy(logits_q, y_qry) + kl_weight * net.kl_reg 


		with torch.no_grad():
			pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
			losses_q[k+1] += sgvlb_loss_q

			correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
			corrects[k + 1] = corrects[k + 1] + correct

	my_losses = [my_loss.cpu().item() for my_loss in losses_q]

	accs = np.array(corrects) /querysz

	return accs,my_losses







