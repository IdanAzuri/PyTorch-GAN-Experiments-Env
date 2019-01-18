import os
import time
from copy import deepcopy

import numpy as np
import torch
from hbconfig import Config
from logger import Logger
from miniimagenet_loader import read_dataset_test, _sample_mini_dataset, _mini_batches, _split_train_test
from one_shot_aug.module import PretrainedClassifier, MiniImageNetModel
from one_shot_aug.utils import AverageMeter, accuracy, mkdir_p
from progress.bar import Bar
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim import lr_scheduler
from utils import saving_config

from . import utils


class OneShotAug():
	model_name = "OneShotAug"
	
	# os.makedirs('images', exist_ok=True)
	
	def __init__(self):
		self._transductive = False
		self.use_cuda = True if torch.cuda.is_available() else False
		self.device = torch.device("cuda" if self.use_cuda else "cpu")
		self.tensorboard = utils.TensorBoard(Config.train.model_dir)
		self.classifier_optimizer = None
		self.num_classes = Config.model.n_classes
		self.num_shots = Config.model.num_smaples_in_shot
		self.inner_batch_size = Config.train.inner_batch_size
		self.inner_iters = Config.train.inner_iters
		self.replacement = Config.train.replacement
		self.meta_batch_size = Config.train.meta_batch_size
		# load model
		# self.classifier.model.load_state_dict(torch.load(os.path.join(self.model_path + str(Config.model.name) + '.t7')))
		if Config.model.type == "known_net":
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
		self.exp_lr_scheduler = lr_scheduler.StepLR(self.classifier_optimizer, step_size=5000, gamma=0.1)
		
		if resume:
			self.prev_meta_step_count, self.classifier, self.classifier_optimizer = utils.load_saved_model(self.model_path, self.classifier, self.classifier_optimizer)
			print(f"Model has been loaded step:{self.prev_meta_step_count}, path:{self.model_path}")
		self.logger = Logger(os.path.join(self.c_path, 'log.txt'), title=self.title)
		self.logger.set_names(['step', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
		return self._train
	
	def _train(self, data_loader):
		train_loader, validation_loader = data_loader
		# switch to train mode
		# if self.classifier.arch.startswith('alexnet') or self.classifier.arch.startswith('vgg'):
		if self.use_cuda:
			self.classifier = torch.nn.DataParallel(self.classifier).to(self.device)
			if torch.cuda.device_count() > 1:
				print("Using ", torch.cuda.device_count(), " GPUs!")
			cudnn.benchmark = True
		print('Total params: %.2fM' % (sum(p.numel() for p in self.classifier.parameters()) / 1000000.0))
		best_loss = 10e7
		best_model_wts = deepcopy(self.classifier.state_dict())
		for meta_epoch in range(self.meta_batch_size):
			while True:
				self.meta_step_count = self.prev_meta_step_count + 1  # init value
				# training
				train_loss, train_acc = self._train_epoch(train_loader)
				if self.meta_step_count % 10:
					# validation
					# Step Verbose & Tensorboard Summary
					if self.meta_step_count % Config.train.verbose_step_count == 0:
						validation_loss, validation_acc, num_correct = self.evaluate_model(validation_loader)
						self._add_summary(self.meta_step_count, {"loss_valid": validation_loss})
						self._add_summary(self.meta_step_count, {"top1_acc_valid": validation_acc})
						self._add_summary(self.meta_step_count, {"accuracy_valid": num_correct/self.num_classes})
						
						train_loss, train_acc, train_num_correct = self.evaluate_model(train_loader)
						self._add_summary(self.meta_step_count, {"loss_train": train_loss})
						self._add_summary(self.meta_step_count, {"top1_acc_train": train_acc})
						self._add_summary(self.meta_step_count, {"accuracy_train": train_num_correct/self.num_classes})

						self.logger.append([self.meta_step_count, self.learning_rate, train_loss, validation_loss, train_acc, validation_acc])
						# deep copy the model
						print(f"step {self.meta_step_count}: update best weights prev_loss{best_loss} new_best_loss{validation_loss}")
						if validation_loss < best_loss:
							best_loss = validation_loss
							best_model_wts = deepcopy(self.classifier.state_dict())
				
				if self.meta_step_count % 10000 == 0:
					torch.save(best_model_wts, os.path.join(self.model_path + str(Config.model.name) + '.t7'))
					print('save!')
				self.prev_meta_step_count = self.meta_step_count
				# update learning rate
				
				self.exp_lr_scheduler.step()
				if self.meta_step_count >= Config.train.meta_iters:
					self.predict(self.loss_criterion)
					break
	
	def _train_epoch(self, train_loader):
		self.classifier.train()
		data_time = AverageMeter()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()
		end = time.time()
		mini_data_set= _sample_mini_dataset(train_loader, self.num_classes, self.num_shots)
		mini_train_loader = _mini_batches(mini_data_set, self.inner_batch_size, self.inner_iters, self.replacement)
		
		for batch_idx, batch in enumerate(mini_train_loader):
			(inputs, labels) = zip(*batch)
			self.meta_step_count = self.prev_meta_step_count + batch_idx + 1  # init value
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
			

		# Save model parameters
		if self.meta_step_count % Config.train.save_checkpoints_steps == 0:
			utils.save_checkpoint(self.meta_step_count, self.model_path, self.classifier, self.classifier_optimizer)
		
		return losses.avg, top1.avg
	
	def evaluate_model(self, dataset, mode="valid"):
		self.classifier.eval()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()
		
		train_set, test_set = _split_train_test(_sample_mini_dataset(dataset, self.num_classes, self.num_shots + 1))  # 1 more sample for train
		old_model_state = deepcopy(self.classifier.state_dict())  # store weights to avoid training
		mini_batches = _mini_batches(train_set, self.inner_batch_size, self.inner_iters, self.replacement)
		for batch_idx, batch in enumerate(mini_batches):
			(inputs, labels) = zip(*batch)
			step_count = self.prev_meta_step_count + batch_idx + 1  # init value
			# measure data loading time
			
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
			# if step_count % Config.train.verbose_step_count == 0:
			# 	self._add_summary(step_count, {f"loss_{mode}": losses.avg})
			# 	self._add_summary(step_count, {f"top1_acc_{mode}": top1.avg})
			# 	self._add_summary(step_count, {f"top5_acc_{mode}": top5.avg})
		test_preds = self._test_predictions(train_set, test_set)  # testing on only 1 sample mabye redundant
		num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
		print(f"step{step_count}| {mode}_loss{losses.avg}| acc{top1.avg}, num_correct: {num_correct}")
		self.classifier.load_state_dict(old_model_state)  # load back model's weights
		if mode == "total_test":
			return num_correct
		return losses.avg, top1.avg, num_correct
	
	def predict(self, criterion):
		print("Predicting on test set...")
		self.loss_criterion = criterion
		
		# Load model
		self.prev_meta_step_count, self.classifier, self.classifier_optimizer = utils.load_saved_model(self.model_path, self.classifier, self.classifier_optimizer)
		print(f"Model has been loaded step:{self.prev_meta_step_count}, path:{self.model_path}")
		if self.use_cuda:
			self.classifier.cuda()
		test_loader = read_dataset_test(Config.data.miniimagenet_path)[0]
		evaluation = self.evaluate(test_loader)
		print(f"Total score: {evaluation}")
		return evaluation
	
	def build_criterion(self):
		return torch.nn.CrossEntropyLoss().to(self.device)
	
	def build_optimizers(self, classifier):
		
		classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=self.learning_rate, betas=Config.train.optim_betas)
		
		return classifier_optimizer
	
	def _add_summary(self, step, summary):
		for tag, value in summary.items():
			self.tensorboard.scalar_summary(tag, value, step)
	
	def _test_predictions(self, train_set, test_set):
		if self._transductive:
			inputs, _ = zip(*test_set)
			if self.use_cuda:
				inputs.cuda()
			return self.classifier(inputs)
		res = []
		for test_sample in test_set:
			inputs, _ = zip(*train_set)
			if self.use_cuda:
				inputs = Variable(torch.stack(inputs)).cuda()
				inputs += Variable(torch.stack((test_sample[0],))).cuda()
			else:
				inputs = Variable(torch.stack(inputs))
				inputs += Variable(torch.stack((test_sample[0],)))
			res.append(torch.argmax(self.classifier(inputs)))
		return res
	
	def evaluate(self, dataset, num_classes=5, num_samples=10000):
		"""
		Evaluate a model on a dataset. Final test!
		"""
		total_correct = 0
		for _ in range(num_samples):
			total_correct += self.evaluate_model(dataset, mode="total_test")
		
		return total_correct.item() / (num_samples * num_classes)

# TODO: finish evaluation correctly like reptile
# maybe redundant 20 inner loop because it the same image
