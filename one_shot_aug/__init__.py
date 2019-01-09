import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
from hbconfig import Config
from progress.bar import Bar
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim import lr_scheduler

from logger import Logger
from miniimagenet_loader import read_dataset_test, _sample_mini_dataset, _mini_batches
from one_shot_aug.module import PretrainedClassifier, MiniImageNetModel
from one_shot_aug.utils import AverageMeter, accuracy, mkdir_p
from utils import saving_config
from . import utils


class OneShotAug():
	model_name = "OneShotAug"
	
	# os.makedirs('images', exist_ok=True)
	
	def __init__(self):
		self.use_cuda = True if torch.cuda.is_available() else False
		self.device = torch.device("cuda" if self.use_cuda else "cpu")
		self.tensorboard = utils.TensorBoard(Config.train.model_dir)
		# load model
		# self.classifier.model.load_state_dict(torch.load(os.path.join(self.model_path + str(Config.model.name) + '.t7')))
		if Config.model.pretrained:
			self.classifier = PretrainedClassifier()
			self.title = self.classifier.title
			self.classifier = self.classifier.model
		else:
			self.classifier = MiniImageNetModel()
			self.title = self.classifier.title
		self.learning_rate = Config.train.learning_rate
		self.c_path = f"{Config.train.model_dir}/log"
		self.model_path = f"{Config.train.model_dir}/model"
		mkdir_p(self.c_path)
		mkdir_p(self.model_path)
		# Print Config setting
		saving_config(os.path.join(self.c_path, "config_log.txt"))
	
	def train_fn(self, criterion, optimizer, resume=True):
		self.loss_criterion = criterion
		self.classifier_optimizer = optimizer
		
		if resume:
			self.prev_meta_step_count, self.classifier, self.classifier_optimizer = utils.load_saved_model(self.model_path, self.classifier, self.classifier_optimizer)
		self.logger = Logger(os.path.join(self.c_path, 'log.txt'), title=self.title)
		self.logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
		return self._train
	
	def _train(self, data_loader):
		train_loader, validation_loader = data_loader
		self.num_classes = Config.model.n_classes
		self.num_shots = Config.model.num_smaples_in_shot
		self.inner_batch_size = Config.train.inner_batch_size
		self.inner_iters = Config.train.inner_iters
		self.replacement = Config.train.replacement
		self.meta_batch_size = Config.train.meta_batch_size
		# switch to train mode
		# if self.classifier.arch.startswith('alexnet') or self.classifier.arch.startswith('vgg'):
		if self.use_cuda:
			self.classifier = torch.nn.DataParallel(self.classifier).to(self.device)
			if torch.cuda.device_count() > 1:
				print("Using ", torch.cuda.device_count(), " GPUs!")
			cudnn.benchmark = True
		print('Total params: %.2fM' % (sum(p.numel() for p in self.classifier.parameters()) / 1000000.0))
		best_loss = 10e10
		best_model_wts = deepcopy(self.classifier.state_dict())
		while True:
			self.meta_step_count = self.prev_meta_step_count + 1  # init value
			for meta_epoch in range(self.meta_batch_size):
				#training
				mini_train_dataset = _sample_mini_dataset(train_loader, self.num_classes, self.num_shots)
				mini_train_batches = _mini_batches(mini_train_dataset, self.inner_batch_size, self.inner_iters, self.replacement)
				train_loss, train_acc = self._train_epoch(mini_train_batches)
				#validation
				mini_valid_dataset = _sample_mini_dataset(validation_loader, self.num_classes, self.num_shots)
				mini_valid_batches = _mini_batches(mini_valid_dataset, self.inner_batch_size, self.inner_iters, self.replacement)
				validation_loss, validation_acc = self.evaluate_model(mini_valid_batches)
				
				self.logger.append([self.learning_rate, train_loss, validation_loss, train_acc, validation_acc])
				epoch_loss = validation_loss
				# deep copy the model
				if epoch_loss < best_loss:
					print(f"Update best weights prev_loss{best_loss} new_best_loss{epoch_loss}")
					best_loss = epoch_loss
					best_model_wts = deepcopy(self.classifier.state_dict())
				
				if meta_epoch % 30 == 0:
					torch.save(best_model_wts, os.path.join(self.model_path + str(Config.model.name) + '.t7'))
			print('save!')
			self.prev_meta_step_count = self.meta_step_count
			# update learning rate
			exp_lr_scheduler = lr_scheduler.StepLR(self.classifier_optimizer, step_size=5000, gamma=0.1)
			exp_lr_scheduler.step()
			if self.meta_step_count >= Config.train.meta_iters:
				self.predict()
				sys.exit()
	
	def _train_epoch(self, train_loader):
		self.classifier.train()
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()
		end = time.time()
		if Config.train.show_progrees_bar:
			bar = Bar('Processing', max=self.num_classes * self.num_shots)
		for batch_idx, batch in enumerate(train_loader):
			(inputs, labels) = zip(*batch)
			step_count = self.prev_meta_step_count + batch_idx + 1  # init value
			# measure data loading time
			data_time.update(time.time() - end)
			
			inputs = Variable(torch.stack(inputs))
			labels = Variable(torch.from_numpy(np.array(labels)))
			if self.use_cuda:
				inputs = inputs.cuda()
				labels = labels.cuda()
			
			# compute output
			outputs = self.classifier(inputs)
			loss = self.loss_criterion(outputs, labels)
			
			# measure accuracy and record loss
			prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
			losses.update(loss.data.item(), inputs.size(0))
			top1.update(prec1.item(), inputs.size(0))
			top5.update(prec5.item(), inputs.size(0))
			
			# compute gradient and do SGD step
			self.classifier_optimizer.zero_grad()
			loss.backward()
			self.classifier_optimizer.step()
			
			# Step Verbose & Tensorboard Summary
			if self.meta_step_count + batch_idx % Config.train.verbose_step_count == 0:
				self._add_summary(self.meta_step_count, {"loss_train": losses.avg})
				self._add_summary(step_count, {"top1_acc_train": top1.avg})
				self._add_summary(step_count, {"top5_acc_train": top5.avg})
			
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			
			# plot progress
			if Config.train.show_progrees_bar:
				bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
					batch=batch_idx + 1, size=len(train_loader), data=data_time.val, bt=batch_time.val, total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg, top1=top1.avg,
					top5=top5.avg)
				bar.next()
		if Config.train.show_progrees_bar:
			bar.finish()
		# Save model parameters
		if self.meta_step_count % Config.train.save_checkpoints_steps == 0:
			utils.save_checkpoint(self.meta_step_count, self.model_path, self.classifier, self.classifier_optimizer)
		
		return losses.avg, top1.avg
	
	def evaluate_model(self, data_loader):
		self.classifier.eval()
		
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()
		end = time.time()
		
		for batch_idx, batch in enumerate(data_loader):
			(inputs, labels) = zip(*batch)
			step_count = self.prev_meta_step_count + batch_idx + 1  # init value
			# measure data loading time
			data_time.update(time.time() - end)
			
			inputs = Variable(torch.stack(inputs))
			labels = Variable(torch.from_numpy(np.array(labels)))
			if self.use_cuda:
				inputs = inputs.cuda()
				labels = labels.cuda()
			
			# compute output
			outputs = self.classifier(inputs)
			loss = self.loss_criterion(outputs, labels)
			
			# measure accuracy and record loss
			prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
			losses.update(loss.data.item(), inputs.size(0))
			top1.update(prec1.item(), inputs.size(0))
			top5.update(prec5.item(), inputs.size(0))
			
			# Step Verbose & Tensorboard Summary
			if step_count % Config.train.verbose_step_count == 0:
				self._add_summary(step_count, {"loss_valid": losses.avg})
				self._add_summary(step_count, {"top1_acc_valid": top1.avg})
				self._add_summary(step_count, {"top5_acc_valid": top5.avg})
		
		return losses.avg, top1.avg
	
	def predict(self):
		print("Predicting on test set...")
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()
		# Load model
		self.prev_meta_step_count, self.classifier, self.classifier_optimizer = utils.load_saved_model(self.model_path, self.classifier, self.classifier_optimizer)
		test_loader = read_dataset_test(Config.data.miniimagenet_path)
		mini_test_dataset = _sample_mini_dataset(test_loader, self.num_classes, self.num_shots)
		mini_test_batches = _mini_batches(mini_test_dataset, self.inner_batch_size, self.inner_iters, self.replacement)
		self.classifier.eval()
		for batch_idx, batch in enumerate(mini_test_batches):
			(inputs, labels) = zip(*batch)
			step_count = self.prev_meta_step_count + batch_idx + 1  # init value
			
			inputs = Variable(torch.stack(inputs))
			labels = Variable(torch.from_numpy(np.array(labels)))
			if self.use_cuda:
				inputs = inputs.cuda()
				labels = labels.cuda()
			
			# compute output
			outputs = self.classifier(inputs)
			loss = self.loss_criterion(outputs, labels)
			
			# measure accuracy and record loss
			prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
			losses.update(loss.data.item(), inputs.size(0))
			top1.update(prec1.item(), inputs.size(0))
			top5.update(prec5.item(), inputs.size(0))
			
			# Step Verbose & Tensorboard Summary
			if step_count % Config.train.verbose_step_count == 0:
				self._add_summary(step_count, {"loss_test": losses.avg})
				self._add_summary(step_count, {"top1_acc_test": top1.avg})
				self._add_summary(step_count, {"top5_acc_test": top5.avg})
		
		return losses.avg, top1.avg
	
	def build_criterion(self):
		return torch.nn.CrossEntropyLoss().to(self.device)
	
	def build_optimizers(self, classifier):
		
		classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=self.learning_rate, betas=Config.train.optim_betas)
		
		return classifier_optimizer
	
	def _add_summary(self, step, summary):
		for tag, value in summary.items():
			self.tensorboard.scalar_summary(tag, value, step)
