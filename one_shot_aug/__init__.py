import os
import sys
import time
from copy import deepcopy

import torch
from hbconfig import Config
from progress.bar import Bar
from torch.backends import cudnn
from torch.optim import lr_scheduler

from logger import Logger
from one_shot_aug.module import PretrainedClassifier
from one_shot_aug.utils import AverageMeter, accuracy, mkdir_p
from . import utils


outerstepsize0 = 0.1  # stepsize of outer optimization, i.e., meta-optimization


class OneShotAug():
	model_name = "OneShotAug"
	os.makedirs('images', exist_ok=True)
	
	def __init__(self):
		self.use_cuda = True if torch.cuda.is_available() else False
		self.device = torch.device("cuda" if self.use_cuda else "cpu")
		self.tensorboard = utils.TensorBoard(Config.train.model_dir)
		self.classifier = PretrainedClassifier()
		self.learning_rate = Config.train.d_learning_rate
		self.c_path = f"{Config.train.model_dir}/classifier"
		mkdir_p(self.c_path)
	
	def train_fn(self, criterion, optimizer, resume=True):
		self.loss_criterion = criterion
		self.classifier_optimizer = optimizer
		
		if resume:
			self.prev_step_count, self.classifier, self.classifier_optimizer = utils.load_saved_model(self.c_path, self.classifier, self.classifier_optimizer)
		self.logger = Logger(os.path.join(self.c_path, 'log.txt'), title=self.classifier.title)
		self.logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
		return self._train
	
	def _train(self, data_loader):
		train_loader, validation_loader = data_loader
		print(len(validation_loader))
		# switch to train mode
		if self.classifier.arch.startswith('alexnet') or self.classifier.arch.startswith('vgg'):
			self.classifier.model.features = torch.nn.DataParallel(self.classifier.model.features)
			self.classifier.model.to(self.device)
		else:
			self.classifier.model = torch.nn.DataParallel(self.classifier.model).to(self.device)
		if self.use_cuda:
			cudnn.benchmark = True
		print('    Total params: %.2fM' % (sum(p.numel() for p in self.classifier.model.parameters()) / 1000000.0))
		while True:
			train_loss, train_acc = self._train_epoch(train_loader)
			test_loss, test_acc = self.evaluate_model(validation_loader)
			# append logger file
			self.logger.append([self.learning_rate, train_loss, test_loss, train_acc, test_acc])
	
	def _train_epoch(self, train_loader):
		global step_count
		# update learning rate
		exp_lr_scheduler = lr_scheduler.StepLR(self.classifier_optimizer, step_size=7, gamma=0.1)
		exp_lr_scheduler.step()
		
		self.classifier.model.train()
		
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()
		end = time.time()
		
		bar = Bar('Processing', max=len(train_loader))
		for batch_idx, (inputs, targets) in enumerate(train_loader):
			step_count = self.prev_step_count + batch_idx + 1  # init value
			# measure data loading time
			data_time.update(time.time() - end)
			
			if self.use_cuda:
				inputs, targets = inputs.cuda(), targets.cuda(async=True)
			inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
			
			# compute output
			outputs = self.classifier.model(inputs)
			loss = self.loss_criterion(outputs, targets)
			
			# measure accuracy and record loss
			prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
			losses.update(loss.data[0], inputs.size(0))
			top1.update(prec1[0], inputs.size(0))
			top5.update(prec5[0], inputs.size(0))
			
			weights_before = deepcopy(self.classifier.state_dict())
			# compute gradient and do SGD step
			self.classifier_optimizer.zero_grad()
			loss.backward()
			self.classifier_optimizer.step()
			
			weights_after = self.classifier.state_dict()
			outerstepsize = outerstepsize0 * (1 - batch_idx / len(train_loader.dataset))  # linear schedule
			self.classifier.load_state_dict({name: weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize for name in weights_before})
			# Step Verbose & Tensorboard Summary
			if step_count % Config.train.verbose_step_count == 0:
				self._add_summary(step_count, {"losses_train": losses.avg})
			
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			
			# plot progress
			bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
				batch=batch_idx + 1, size=len(train_loader), data=data_time.val, bt=batch_time.val, total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg, top1=top1.avg,
				top5=top5.avg)
			bar.next()
		bar.finish()
		# Save model parameters
		if step_count % Config.train.save_checkpoints_steps == 0:
			utils.save_checkpoint(step_count, self.c_path, self.classifier, self.classifier_optimizer)
		self.prev_step_count = step_count
		if step_count >= Config.train.train_steps:
			sys.exit()
		return losses.avg, top1.avg
	
	def evaluate_model(self, data_loader):
		self.classifier.model.eval()
		
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()
		end = time.time()
		
		bar = Bar('Processing', max=len(data_loader))
		for batch_idx, (inputs, targets) in enumerate(data_loader):
			step_count = self.prev_step_count + batch_idx + 1  # init value
			# measure data loading time
			data_time.update(time.time() - end)
			
			if self.use_cuda:
				inputs, targets = inputs.cuda(), targets.cuda(async=True)
			inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
			
			# compute output
			outputs = self.classifier.model(inputs)
			loss = self.loss_criterion(outputs, targets)
			
			# measure accuracy and record loss
			prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
			losses.update(loss.data[0], inputs.size(0))
			top1.update(prec1[0], inputs.size(0))
			top5.update(prec5[0], inputs.size(0))
			
			# Step Verbose & Tensorboard Summary
			if step_count % Config.train.verbose_step_count == 0:
				self._add_summary(step_count, {"loss_valid": losses.avg})
			
			# plot progress
			bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
				batch=batch_idx + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg, total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg, top1=top1.avg,
				top5=top5.avg, )
			bar.next()
		bar.finish()
		return losses.avg, top1.avg
	
	def predict(self):
		pass
	
	def build_criterion(self):
		return torch.nn.CrossEntropyLoss().to(self.device)
	
	def save_model_params(self, model):
		self.weights_before = deepcopy(model.state_dict())
		# torch.save(model.state_dict(), path)
		return self.weights_before
	
	def build_optimizers(self, classifier):
		
		classifier_optimizer = torch.optim.Adam(classifier.model.parameters(), lr=self.learning_rate, betas=Config.train.optim_betas)
		
		return classifier_optimizer
	
	def _add_summary(self, step, summary):
		for tag, value in summary.items():
			self.tensorboard.scalar_summary(tag, value, step)
