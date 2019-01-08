import os
import random
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
from one_shot_aug.module import PretrainedClassifier, MiniImageNetModel
from one_shot_aug.utils import AverageMeter, accuracy, mkdir_p
from utils import saving_config
from . import utils


outerstepsize0 = 0.1  # stepsize of outer optimization, i.e., meta-optimization


class OneShotAug():
	model_name = "OneShotAug"
	os.makedirs('images', exist_ok=True)
	
	def __init__(self):
		self.use_cuda = True if torch.cuda.is_available() else False
		self.device = torch.device("cuda" if self.use_cuda else "cpu")
		self.tensorboard = utils.TensorBoard(Config.train.model_dir)
		self.classifier = MiniImageNetModel()
		self.learning_rate = Config.train.learning_rate
		self.c_path = f"{Config.train.model_dir}/classifier"
		mkdir_p(self.c_path)
		# Print Config setting
		saving_config(os.path.join(self.c_path, "config_log.txt"))
	
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
		self.num_classes = Config.model.n_classes
		self.num_shots = Config.model.num_smaples_in_shot
		self.inner_batch_size = Config.train.inner_batch_size
		self.inner_iters = Config.train.inner_iters
		self.replacement = Config.train.replacement
		self.meta_batch_size = Config.train.meta_batch_size
		# switch to train mode
		# if self.classifier.arch.startswith('alexnet') or self.classifier.arch.startswith('vgg'):
		if self.use_cuda:
			cudnn.benchmark = True
			self.classifier = torch.nn.DataParallel(self.classifier).to(self.device)
			if torch.cuda.device_count() > 1:
				print("Let's use", torch.cuda.device_count(), "GPUs!")
		print('    Total params: %.2fM' % (sum(p.numel() for p in self.classifier.parameters()) / 1000000.0))
		while True:
			self.step_count = self.prev_step_count + 1  # init value
		
			for _ in range(self.meta_batch_size):
					mini_train_dataset = _sample_mini_dataset(train_loader, self.num_classes, self.num_shots)
					mini_train_batches = _mini_batches(mini_train_dataset, self.inner_batch_size, self.inner_iters, self.replacement)
					train_loss, train_acc = self._train_epoch(mini_train_batches)
					mini_valid_dataset = _sample_mini_dataset(validation_loader, self.num_classes, self.num_shots)
					mini_valid_batches = _mini_batches(mini_valid_dataset, self.inner_batch_size, self.inner_iters, self.replacement)
					test_loss, test_acc = self.evaluate_model(mini_valid_batches)
					# append logger file
					self.logger.append([self.learning_rate, train_loss, test_loss, train_acc, test_acc])
			self.prev_step_count = self.step_count
			if self.step_count >= Config.train.meta_iters:
				sys.exit()
	def _train_epoch(self, train_loader):
		# update learning rate
		exp_lr_scheduler = lr_scheduler.StepLR(self.classifier_optimizer, step_size=7, gamma=0.1)
		exp_lr_scheduler.step()
		
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
			# measure data loading time
			data_time.update(time.time() - end)
			
			if self.use_cuda:
				inputs = Variable(torch.from_numpy(np.asarray(inputs).reshape(self.inner_batch_size, Config.data.channels, Config.data.image_size, Config.data.image_size))).cuda()
				labels = Variable(torch.from_numpy(np.array(labels))).cuda()
			else:
				inputs = Variable(torch.from_numpy(np.asarray(inputs).reshape(self.inner_batch_size, Config.data.channels, Config.data.image_size, Config.data.image_size)))
				labels = Variable(torch.from_numpy(np.array(labels)))
			# compute output
			outputs = self.classifier(inputs)
			loss = self.loss_criterion(outputs, labels)
			
			# measure accuracy and record loss
			prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
			losses.update(loss.data.item(), inputs.size(0))
			top1.update(prec1.item(), inputs.size(0))
			top5.update(prec5.item(), inputs.size(0))
			
			# weights_before = deepcopy(self.classifier.model.state_dict())
			# compute gradient and do SGD step
			self.classifier_optimizer.zero_grad()
			loss.backward()
			self.classifier_optimizer.step()
			
			# weights_after = self.classifier.model.state_dict()
			# outerstepsize = outerstepsize0 * (1 - batch_idx / len(train_loader.dataset))  # linear schedule
			# self.classifier.model.load_state_dict({name: weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize for name in weights_before})
			# Step Verbose & Tensorboard Summary
			if self.step_count % Config.train.verbose_step_count == 0:
				self._add_summary(self.step_count, {"loss_train": losses.avg})
			
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			
			# plot progress
			if Config.train.show_progrees_bar:
				bar.suffix = '({batch}/{size}) Data: {data:} | Batch: {bt:} | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
					batch=batch_idx + 1, size=self.num_classes*self.num_shots, data=data_time.val, bt=batch_time.val, total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg, top1=top1.avg,
					top5=top5.avg)
				bar.next()
		if Config.train.show_progrees_bar:
			bar.finish()
		# Save model parameters
		if self.step_count % Config.train.save_checkpoints_steps == 0:
			utils.save_checkpoint(self.step_count, self.c_path, self.classifier, self.classifier_optimizer)
		
		return losses.avg, top1.avg
	
	def evaluate_model(self, data_loader):
		self.classifier.eval()
		
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()
		end = time.time()
		
		bar = Bar('Processing', max=self.num_classes * self.num_shots)
		for batch_idx, batch in enumerate(data_loader):
			(inputs, labels) = zip(*batch)
			step_count = self.prev_step_count + batch_idx + 1  # init value
			# measure data loading time
			data_time.update(time.time() - end)
			
			if self.use_cuda:
				inputs = Variable(torch.from_numpy(np.asarray(inputs).reshape(Config.model.n_classes, Config.data.channels, Config.data.image_size, Config.data.image_size))).cuda()
				labels = Variable(torch.from_numpy(np.array(labels))).cuda()
			else:
				inputs = Variable(torch.from_numpy(np.asarray(inputs).reshape(Config.model.n_classes, Config.data.channels, Config.data.image_size, Config.data.image_size)))
				labels = Variable(torch.from_numpy(np.array(labels)))
			
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
			
			# plot progress
			bar.suffix = '({batch}/{size}) Data: {data:} | Batch: {bt:} | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
				batch=batch_idx + 1, size=self.num_classes*self.num_shots, data=data_time.avg, bt=batch_time.avg, total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg, top1=top1.avg,
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
		
		classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=self.learning_rate, betas=Config.train.optim_betas)
		
		return classifier_optimizer
	
	def _add_summary(self, step, summary):
		for tag, value in summary.items():
			self.tensorboard.scalar_summary(tag, value, step)


def _sample_mini_dataset(dataset, num_classes, num_shots):
	"""
	Sample a few shot task from a dataset.
	Returns:
	  An iterable of (input, label) pairs.
	"""
	shuffled = list(dataset)
	random.shuffle(shuffled)
	for class_idx, class_obj in enumerate(shuffled[:num_classes]):
		for sample in class_obj.sample(num_shots):
			yield (sample, class_idx)


def _mini_batches(samples, batch_size, num_batches, replacement):
	"""
	Generate mini-batches from some data.
	Returns:
	  An iterable of sequences of (input, label) pairs,
		where each sequence is a mini-batch.
	"""
	samples = list(samples)
	if replacement:
		for _ in range(num_batches):
			yield random.sample(samples, batch_size)
		return
	cur_batch = []
	batch_count = 0
	while True:
		random.shuffle(samples)
		for sample in samples:
			cur_batch.append(sample)
			if len(cur_batch) < batch_size:
				continue
			yield cur_batch
			cur_batch = []
			batch_count += 1
			if batch_count == num_batches:
				return


def _split_train_test(samples, test_shots=1):
	"""
	Split a few-shot task into a train and a test set.
	Args:
	  samples: an iterable of (input, label) pairs.
	  test_shots: the number of examples per class in the
		test set.
	Returns:
	  A tuple (train, test), where train and test are
		sequences of (input, label) pairs.
	"""
	train_set = list(samples)
	test_set = []
	labels = set(item[1] for item in train_set)
	for _ in range(test_shots):
		for label in labels:
			for i, item in enumerate(train_set):
				if item[1] == label:
					del train_set[i]
					test_set.append(item)
					break
	if len(test_set) < len(labels) * test_shots:
		raise IndexError('not enough examples of each class for test set')
	return train_set, test_set
