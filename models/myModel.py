import torch
import torchvision.models as models
import torch.nn as nn
import inspect
import re
import numpy as np 
import torch.nn.functional as F 


class MyModel(nn.Module):

	def __init__(self,model_config, threshold, init_linsvdo = -8, init_convsvdo = -8, pretrained=False, svdo=False, custom=True):

		super(MyModel,self).__init__()

		if custom:
			self.config = model_config
			self.pretrained_feat_params = [] # replace if using pretrained ; feature NA as of now
			if pretrained:         # keep as False
				self.pretrained_feat_params = getCustomPretrained(pretrained)
		# else:

		# 	self.config,self.pretrained_feat_params = getVGGConfig(model_config,pretrained,svdo)

		print('PRETRAINED',len(self.pretrained_feat_params))
		# print(self.config)
		self.pretrained = pretrained
		self.svdo = svdo
		self.model_config = model_config

		self.init_linsvdo = init_linsvdo
		self.init_convsvdo = init_convsvdo
		self.threshold = threshold
		self.model_layers = []

		# this dict contains all tensors needed to be optimized
		self.vars = nn.ParameterList()
		# running_mean and running_var
		self.vars_bn = nn.ParameterList()
		k = 0
		print(self.config)
		for i, (name, param) in enumerate(self.config):
			if name is 'Conv2d':
				# gain=1 according to cbfin's implementation
				if self.pretrained:
					self.vars.append(self.pretrained_feat_params[k][0])       # weight
					self.vars.append(self.pretrained_feat_params[k][1])       # bias
					k+=1
				else:
					# [ch_out, ch_in, kernelsz, kernelsz]
					w = nn.Parameter(torch.ones(*param[:4]))
					torch.nn.init.kaiming_normal_(w)
					self.vars.append(w)
					# [ch_out]
					self.vars.append(nn.Parameter(torch.zeros(param[0])))

				self.model_layers.append('Conv2d')


			elif name is 'Linear':
				# [ch_out, ch_in]
				w = nn.Parameter(torch.ones(*param))
				# gain=1 according to cbfinn's implementation
				torch.nn.init.kaiming_normal_(w)
				self.vars.append(w)
				# [ch_out]
				self.vars.append(nn.Parameter(torch.zeros(param[0])))
				self.model_layers.append('Linear')

			elif name is 'Conv2d_SVDO':
				# [ch_out, ch_in, kernelsz, kernelsz]
				mu  = nn.Parameter(torch.Tensor(*param[:4]))
				log_sigma = nn.Parameter(torch.Tensor(*param[:4]))
				bias = nn.Parameter(torch.Tensor(1,param[0],1,1))

				if self.pretrained:
					mu = self.pretrained_feat_params[k][0]
					bias = self.pretrained_feat_params[k][1]
					k+=1
				else:
					bias.data.zero_()
					torch.nn.init.kaiming_normal_(mu)

					# mu.data.normal_(0, 0.02)
				log_sigma.data.fill_(self.init_convsvdo)


				self.vars.append(mu)
				self.vars.append(bias)
				self.vars.append(log_sigma)


			elif name is 'Linear_SVDO':
				mu = nn.Parameter(torch.Tensor(param[0],param[1])) # torch.nn.parameter.Parameter of size out_features x in_features
				log_sigma = nn.Parameter(torch.Tensor(param[0],param[1]))# torch.nn.parameter.Parameter of size out_features x in_features
				bias = nn.Parameter(torch.Tensor(1,param[0]))

				if self.pretrained:
					print('hi')
					mu = self.pretrained_feat_params[k][0]
					bias = self.pretrained_feat_params[k][1]
					k+=1
				else:
					bias.data.zero_()
					torch.nn.init.kaiming_normal_(mu)

				# mu.data.normal_(0, 0.02)
				log_sigma.data.fill_(self.init_linsvdo)

				self.vars.append(mu)
				self.vars.append(bias)
				self.vars.append(log_sigma)

			elif name is 'bn':
				# [ch_out]
				w = nn.Parameter(torch.ones(param[0]))
				self.vars.append(w)
				# [ch_out]    for step in range(500):

				self.vars.append(nn.Parameter(torch.zeros(param[0])))

				# must set requires_grad=False
				running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
				running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
				self.vars_bn.extend([running_mean, running_var])
				self.model_layers.append('bn')

			elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'MaxPool2d',
						  'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
				self.model_layers.append(name)
				continue
			else:
				raise NotImplementedError


	def forward(self, x, vars=None, bn_training=True):

		if vars is None:
			vars = self.vars
		idx = 0
		bn_idx = 0
		self.kl_reg = 0.0
		self.conv_idx,self.lin_idx=0,0
		self.sp_layer_names = []
		self.sp_vals = []
		self.dropout_vals = {}


		for ii,(name, param) in enumerate(self.config):
			if name is 'Conv2d':
				w, b = vars[idx], vars[idx + 1]
				# remember to keep synchrozied of forward_encoder and forward_decoder!
				# print(ii,x.shape)
				x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
				idx += 2
				# print(name, param, '\tout:', x.shape)
			elif name is 'convt2d':
				w, b = vars[idx], vars[idx + 1]
				# remember to keep synchrozied of forward_encoder and forward_decoder!
				x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
				idx += 2
				# print(name, param, '\tout:', x.shape)


			elif name is 'Conv2d_SVDO' or name is 'Linear_SVDO':
				mu, b, log_sigma = vars[idx],vars[idx+1],vars[idx+2]

				# print('MU NORM',mu.data.norm(2))
				log_alpha = log_sigma * 2.0 - 2.0 * torch.log(1e-8 + torch.abs(mu))
				log_alpha = torch.clamp(log_alpha, -10, 10)

				if torch.isnan(mu).any():
					print('mu',name)

				if torch.isnan(log_sigma).any():
					print('log_sigma',name)

				if torch.isnan(log_alpha).any():
					print('Log Alpha',name)
				if torch.isnan(x).any():
					print('Name',name)
					print(x)
					break

				self.kl_reg += self.kl_reg_term(log_alpha)

				sp_val = (log_alpha.cpu().data.numpy() > self.threshold).mean()
				self.sp_vals.append(sp_val)
				if self.training:
					if name is 'Conv2d_SVDO':
						layer_name = 'Conv-'+str(self.conv_idx+1)
						self.sp_layer_names.append(layer_name)
						self.dropout_vals[layer_name] = log_alpha.cpu().data.numpy()
						
						self.conv_idx+=1
						lrt_mean = F.conv2d(x,mu,bias=None, stride=param[4], padding=param[5])+b
						lrt_std = torch.sqrt(F.conv2d(x*x,torch.exp(log_sigma*2.0)+1e-8,bias=None, stride=param[4], padding=param[5]))
					else:
						layer_name = 'Linear-'+str(self.lin_idx+1)
						self.sp_layer_names.append(layer_name)
						self.dropout_vals[layer_name] = log_alpha.cpu().data.numpy()

						self.lin_idx+=1
						lrt_mean = F.linear(x,mu)+ b # compute mean activation using LRT
						lrt_std = torch.sqrt(F.linear(x*x,torch.exp(log_sigma*2.0)+1e-18))
				
					eps = lrt_std.data.new(lrt_std.size()).normal_()
					x = lrt_mean+lrt_std*eps
				else:
					if name is 'Conv2d_SVDO':
						x = F.conv2d(x, mu * (log_alpha < self.threshold).float(),bias=None, stride=param[4], padding=param[5]) + b
					else:
						x = F.linear(x, mu * (log_alpha < self.threshold).float()) + b
				idx+=3

			elif name is 'Linear':
				w, b = vars[idx], vars[idx + 1]
				# print(w.shape,b.shape,x.shape)
				x = F.linear(x, w, b)
				idx += 2
				# print('forward:', idx, x.norm().item())
			elif name is 'bn':
				w, b = vars[idx], vars[idx + 1]
				running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
				x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
				idx += 2
				bn_idx += 2

			elif name is 'flatten':
				# print('Flatten',x.shape)
				x = x.view(x.size(0), -1)
			elif name is 'reshape':
				# [b, 8] => [b, 2, 2, 2]
				x = x.view(x.size(0), *param)
			elif name is 'relu':
				x = F.relu(x, inplace=param[0])
			elif name is 'leakyrelu':
				x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
			elif name is 'tanh':
				x = F.tanh(x)
			elif name is 'sigmoid':
				x = torch.sigmoid(x)
			elif name is 'upsample':
				x = F.upsample_nearest(x, scale_factor=param[0])
			elif name is 'MaxPool2d':
				x = F.max_pool2d(x, param[0], param[1], param[2])
			elif name is 'avg_pool2d':
				x = F.avg_pool2d(x, param[0], param[1], param[2])

			else:
				print(name)
				raise NotImplementedError

		# make sure variable is used properly
		assert idx == len(vars)
		assert bn_idx == len(self.vars_bn)


		return x

	def zero_grad(self, vars=None):
		"""

		:param vars:
		:return:
		"""
		with torch.no_grad():
			if vars is None:
				for p in self.vars:
					if p.grad is not None:
						p.grad.zero_()
			else:
				for p in vars:
					if p.grad is not None:
						p.grad.zero_()

	def parameters(self):
		"""
		override this function since initial parameters will return with a generator.
		:return:
		"""
		return self.vars

	def get_model_sparsity(self):
		return self.sp_layer_names,self.sp_vals,self.dropout_vals
		
	def set_param_vals(self):
		pass

	def kl_reg_term(self,log_alpha):
		k1, k2, k3 = torch.Tensor([0.63576]).cuda(), torch.Tensor([1.8732]).cuda(), torch.Tensor([1.48695]).cuda()
		# kl is a scalar torch.Tensor 
		kl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha))# eval KL using the approximation
		kl = -torch.sum(kl)
		return kl


# if __name__=='__main__':
# 	model = MyModel('vgg11_bn',threshold=3,pretrained=False)
# 	# print(list(model.children()))
# 	# print(model.num_layers)