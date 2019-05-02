import copy
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from hbconfig import Config
from torch.autograd import Variable
from torch.nn import init

from data_loader import get_loader
from utils import _conv_layer, find_latest, get_sorted_path, mkdir_p


class PretrainedClassifier(nn.Module):
	def __init__(self):
		# create model
		super().__init__()
		arch = Config.model.arch
		self.arch = arch
		self.path_to_save = f"{Config.train.model_dir}/model"
		print("=> creating model '{}'".format(arch))
		print("=> using pre-trained model '{}'".format(arch))
		# model = models.__dict__[arch](pretrained=True)
		if arch.startswith('densenet') or arch.startswith('inception'):
			model_conv = models.__dict__[arch](pretrained=Config.model.pretrained)
			for param in model_conv.parameters():
				param.requires_grad = False
			num_ftrs = model_conv.classifier.in_features
			model_conv.classifier = nn.Linear(num_ftrs, Config.model.n_classes)
			self.model = model_conv
		elif arch.startswith('vgg'):
			# model_conv = models.__dict__[arch](pretrained=Config.model.pretrained)
			model_conv = models.__dict__[arch](False)
			epoch, classifier = self.load_saved_model(self.path_to_save, model_conv)
			self.model = classifier
			print(f"{arch} has been loaded epoch:{epoch}, path:{self.path_to_save}")
			# Number of filters in the bottleneck layer
			num_ftrs = model_conv.classifier[6].in_features
			# convert all the layers to list and remove the last one
			features = list(model_conv.classifier.children())[:-1]
			for param in model_conv.parameters():
				param.requires_grad = False
			## Add the last layer based on the num of classes in our dataset
			features.extend([nn.Linear(num_ftrs, Config.model.n_classes)])
			## convert it into container and add it to our model class.
			model_conv.classifier = nn.Sequential(*features)
			model_conv.num_classes = Config.model.n_classes
			
			self.model = model_conv
		elif arch.startswith('resne'):
			# model = models.__dict__[arch](pretrained=Config.model.pretrained)
			epoch, classifier = self.load_saved_model(self.path_to_save, self.model)
			model = classifier
			print(f"Model has been loaded epoch:{epoch}, path:{self.path_to_save}")
			for param in model.parameters():
				param.requires_grad = False
			num_ftrs = model.fc.in_features
			model.fc = nn.Linear(num_ftrs, Config.model.n_classes)
			self.model = model
		
		self.title = Config.model.dataset + '-' + arch
	
	def forward(self, inputs):
		outputs = self.model(inputs)
		
		return outputs
	
	def clone(self, use_cuda):
		clone = PretrainedClassifier()
		clone.model.load_state_dict(self.model.state_dict())
		if use_cuda:
			clone.cuda()
		return clone
	
	def train_model(self):
		since = time.time()
		
		self.use_cuda = True if torch.cuda.is_available() else False
		best_model = self.model
		best_acc = 0.0
		num_epochs = Config.train.epochs
		self.optimizer = get_optimizer(self.model)
		criterion = nn.CrossEntropyLoss()
		mkdir_p(self.path_to_save)
		if self.use_cuda:
			criterion.cuda()
		resume = True
		if resume:
			epoch, classifier = self.load_saved_model(self.path_to_save, self.model)
			self.model = classifier
			print(f"Model has been loaded epoch:{epoch}, path:{self.path_to_save}")
		else:
			epoch = 0
		for epoch in range(num_epochs):
			print('Epoch {}/{}'.format(epoch, num_epochs - 1))
			print('-' * 10)
			
			# Each epoch has a training and validation phase
			for phase in ['train', 'val']:
				if phase == 'train':
					mode = 'train'
					self.optimizer = self.exp_lr_scheduler(self.optimizer, epoch, init_lr=Config.train.learning_rate)
					self.model.train()  # Set model to training mode
				else:
					self.model.eval()
					mode = 'val'
				
				running_loss = 0.0
				running_corrects = 0
				train_dataset, valid_dataset = get_loader("train")
				dset_loaders = {"train": train_dataset, "val": valid_dataset}
				dset_sizes = {x: len(dset_loaders[x]) for x in ['train', 'val']}
				counter = 0
				# Iterate over data.
				for data in dset_loaders[phase]:
					inputs, labels = data
					# print(inputs.size())
					# print(labels)
					# wrap them in Variable
					if self.use_cuda:
						try:
							inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())
							self.model.cuda()
						except:
							print(inputs, labels)
					else:
						inputs, labels = Variable(inputs), Variable(labels)
					
					# Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
					self.optimizer.zero_grad()
					outputs = self.model(inputs)
					_, preds = torch.max(outputs.data, 1)
					
					loss = criterion(outputs, labels)
					# print('loss done')
					# Just so that you can keep track that something's happening and don't feel like the program isn't running.
					# if counter%10==0:
					#     print("Reached iteration ",counter)
					counter += 1
					
					# backward + optimize only if in training phase
					if phase == 'train':
						# print('loss backward')
						loss.backward()
						# print('done loss backward')
						self.optimizer.step()
					# print('done optim')
					# print evaluation statistics
					try:
						# running_loss += loss.data[0]
						running_loss += loss.item()
						# print(labels.data)
						# print(preds)
						running_corrects += torch.sum(preds == labels.data)
					# print('running correct =',running_corrects)
					except:
						print('unexpected error, could not calculate loss or do a sum.')
				print('trying epoch loss')
				epoch_loss = running_loss / dset_sizes[phase]
				epoch_acc = running_corrects.item() / float(dset_sizes[phase])
				print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
				
				# deep copy the model
				if phase == 'val':
					if epoch_acc > best_acc:
						best_acc = epoch_acc
						best_model = copy.deepcopy(self.model)
						print('new best accuracy = ', best_acc)
			self.save_checkpoint(f"train_{epoch}")
			print(f"Saved in {self.path_to_save}")
		time_elapsed = time.time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		print('Best val Acc: {:4f}'.format(best_acc))
		print('returning and looping back')
		return best_model
	
	# This function changes the learning rate over the training model.
	@staticmethod
	def exp_lr_scheduler(optimizer, epoch, init_lr=0.0001, lr_decay_epoch=100):
		"""Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
		lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
		
		if epoch % lr_decay_epoch == 0:
			print('LR is set to {}'.format(lr))
		
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		
		return optimizer
	
	def load_saved_model(self, path, model):
		latest_path = find_latest(path + "/")
		if latest_path is None:
			return 0, model
		
		checkpoint = torch.load(latest_path)
		
		step_count = checkpoint['step_count']
		state_dict = checkpoint['net']
		# if dataparallel
		# if "module" in list(state_dict.keys())[0]:
		try:
			model.load_state_dict(checkpoint['net'])
		except:
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				name = k[6:]  # remove 'module.' of dataparallel
				new_state_dict[name] = v
			
			model.load_state_dict(new_state_dict)
			# else:
		
		print(f"Load checkpoints...! {latest_path}")
		return step_count, model
	
	def save_checkpoint(self, step, max_to_keep=3):
		sorted_path = get_sorted_path(self.path_to_save)
		for i in range(len(sorted_path) - max_to_keep):
			os.remove(sorted_path[i])
		
		full_path = os.path.join(self.path_to_save, f"classifier_{step}.pkl")
		torch.save({"step_count": step, 'net': self.state_dict(), 'optimizer': self.optimizer.state_dict(), }, full_path)


class MiniImageNetModel(nn.Module):
	"""
	A model for Mini-ImageNet classification.
	"""
	
	def __init__(self):
		super(MiniImageNetModel, self).__init__()
		self.arch = "vgg_small"
		self.title = Config.model.dataset + '-' + self.arch
		self.n_filters = Config.model.filters
		# The height and width of downsampled image
		ds_size = Config.data.image_size // 2 ** 4
		
		self.layer1 = _conv_layer(Config.data.channels, self.n_filters, 3)
		self.layer2 = _conv_layer(self.n_filters, self.n_filters, 3)
		self.layer3 = _conv_layer(self.n_filters, self.n_filters, 3)
		self.layer4 = _conv_layer(self.n_filters, self.n_filters, 3)
		self.out = nn.Sequential(nn.Linear(self.n_filters * ds_size ** 2, Config.model.n_classes))
		# self.out = nn.Linear(self.n_filters * ds_size ** 2, Config.model.n_classes)
		
		# Initialize layers
		self.weights_init(self.layer1)
		self.weights_init(self.layer2)
		self.weights_init(self.layer3)
		self.weights_init(self.layer4)
	
	def weights_init(self, module):
		for m in module.modules():
			if isinstance(m, nn.Conv2d):
				init.xavier_uniform_(m.weight, gain=np.sqrt(2))
				init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
	
	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.view(x.size(0), -1)
		x = self.out(x)
		return x
	
	def point_grad_to(self, target_param, is_cuda, step_size):
		'''
		from reptile
		Set .grad attribute of each parameter to be proportional
		to the difference between self and target
		'''
		for p, target_p in zip(self.parameters(), target_param.parameters()):
			if p.grad is None:
				if is_cuda:
					p.grad = Variable(torch.zeros(p.size())).cuda()
				else:
					p.grad = Variable(torch.zeros(p.size()))
			p.grad.data.zero_()  # not sure this is required
			p.grad.data.add_(step_size * (p.data - target_p.data))
	
	def clone(self, use_cuda):
		clone = MiniImageNetModel()
		clone.load_state_dict(self.state_dict())
		if use_cuda:
			clone.cuda()
		return clone


def get_optimizer(net, state=None):
	optimizer = torch.optim.Adam(net.parameters(), lr=Config.train.learning_rate, betas=(Config.train.optim_betas))
	if state is not None:
		optimizer.load_state_dict(state)
	return optimizer
