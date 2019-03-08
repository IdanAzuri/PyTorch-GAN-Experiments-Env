import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
from hbconfig import Config
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim import lr_scheduler

from logger import Logger
from miniimagenet_loader import read_dataset_test, _sample_mini_dataset, _mini_batches, _split_train_test
from one_shot_aug.module import PretrainedClassifier, MiniImageNetModel
from one_shot_aug.utils import AverageMeter, accuracy, mkdir_p
from utils import saving_config
from . import utils


meta_step_size = 0.1  # stepsize of outer optimization, i.e., meta-optimization


class OneShotAug():
	model_name = "OneShotAug"
	
	# os.makedirs('images', exist_ok=True)
	
	def __init__(self):
		self._transductive = Config.model.transductive
		self.use_cuda = True if torch.cuda.is_available() else False
		self.device = torch.device("cuda" if self.use_cuda else "cpu")
		self.tensorboard = utils.TensorBoard(Config.train.model_dir)
		self.classifier_optimizer = None
		self.num_classes = Config.model.n_classes
		self.train_shot = Config.model.train_shot
		self.num_shots = Config.model.num_smaples_in_shot
		self.inner_batch_size = Config.train.inner_batch_size
		self.inner_iters = Config.train.inner_iters
		self.replacement = Config.train.replacement
		self.meta_batch_size = Config.train.meta_batch_size
		self.meta_step_count = 0
		# load model
		# self.classifier.model.load_state_dict(torch.load(os.path.join(self.model_path + str(Config.model.name) + '.t7')))
		if Config.model.type == "known_net":
			self.net = PretrainedClassifier()
			self.title = self.net.title
			self.net = self.net.model
		else:
			self.net = MiniImageNetModel()
			self.title = self.net.title
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
			self.prev_meta_step_count, self.net, self.classifier_optimizer = utils.load_saved_model(self.model_path, self.net, self.classifier_optimizer)
			print(f"Model has been loaded step:{self.prev_meta_step_count}, path:{self.model_path}")
		self.logger = Logger(os.path.join(self.c_path, 'log.txt'), title=self.title)
		self.logger.set_names(['step', 'Learning Rate', 'Train Acc.', 'Valid Acc.'])
		return self._train
	
	def _train(self, data_loader):
		train_loader, validation_loader = data_loader
		# switch to train mode
		# if self.classifier.arch.startswith('alexnet') or self.classifier.arch.startswith('vgg'):
		if self.use_cuda:
			self.net = torch.nn.DataParallel(self.net).to(self.device)
			if torch.cuda.device_count() > 1:
				print("Using ", torch.cuda.device_count(), " GPUs!")
			cudnn.benchmark = True
		print('Total params: %.2fK' % (sum(p.numel() for p in self.net.parameters()) / 1000.0))
		while True:
			self.meta_step_count = self.prev_meta_step_count + 1  # init value
			self.prev_meta_step_count = self.meta_step_count
			# training
			self._train_step(train_loader)
			state = self.classifier_optimizer.state_dict()  # save optimizer state
			
			# validation
			if self.meta_step_count % 10 == 0:
				# Step Verbose & Tensorboard Summary
				if self.meta_step_count % Config.train.verbose_step_count == 0:
					validation_num_correct, valid_count = self.evaluate_model(validation_loader)
					# self._add_summary(self.meta_step_count, {"loss_valid": validation_loss})
					# self._add_summary(self.meta_step_count, {"top1_acc_valid": validation_acc})
					# valid_acc_eval = validation_num_correct / self.num_classes
					valid_acc_eval = float(validation_num_correct) / valid_count
					self._add_summary(self.meta_step_count, {"accuracy_valid": valid_acc_eval})
					
					train_num_correct,train_count = self.evaluate_model(train_loader)
					# self._add_summary(self.meta_step_count, {"loss_train": train_loss})
					# self._add_summary(self.meta_step_count, {"top1_acc_train": train_acc})
					# train_acc_eval = train_num_correct / self.num_classes
					train_acc_eval = float(train_num_correct) / train_count
					self._add_summary(self.meta_step_count, {"accuracy_train": train_acc_eval})
					print(f"step{self.meta_step_count}| accuracy_train: {train_acc_eval}| accuracy_valid:{valid_acc_eval}")
					# loading back optimizer state
					self.classifier_optimizer.load_state_dict(state)
					self.logger.append([self.meta_step_count, self.learning_rate, train_acc_eval,
					                    valid_acc_eval])  # deep copy the model  # print(f"step {self.meta_step_count}:_loss{validation_loss}")
				# Interpolate between current weights and trained weights from this task
				# I.e. (weights_before - weights_after) is the meta-gradient
				
				# if validation_loss < best_loss:
				# 	best_loss = validation_loss
				# 	best_model_wts = deepcopy(self.classifier.state_dict())
				
				# if self.meta_step_count % 10000 == 0:
				# 	torch.save(best_model_wts, os.path.join(self.model_path + str(Config.model.name) + '.t7'))
				# 	print('save!')
				
				# update learning rate
				
				# self.exp_lr_scheduler.step()
				if self.meta_step_count >= Config.train.meta_iters:
					self.predict(self.loss_criterion)
					sys.exit()
	
	def _train_step(self, train_loader):
		self.net.train()
		data_time = AverageMeter()
		# losses = AverageMeter()
		# top1 = AverageMeter()
		# top5 = AverageMeter()
		end = time.time()
		weights_original = deepcopy(self.net.state_dict())
		new_weights = []
		for _ in range(self.meta_batch_size):
			mini_data_set = _sample_mini_dataset(train_loader, self.num_classes, self.train_shot)
			mini_train_loader = _mini_batches(mini_data_set, self.inner_batch_size, self.inner_iters, self.replacement)
			
			for batch_idx, batch in enumerate(mini_train_loader):
				(inputs, labels) = zip(*batch)
				self.meta_step_count = self.prev_meta_step_count + 1  # init value
				# measure data loading time
				data_time.update(time.time() - end)
				
				inputs = Variable(torch.stack(inputs))
				labels = Variable(torch.from_numpy(np.array(labels)))
				if self.use_cuda:
					inputs = inputs.cuda()
					labels = labels.cuda()
				
				# compute output
				outputs = self.net(inputs)
				loss = self.loss_criterion(outputs, labels)
				
				# # measure accuracy and record loss
				# prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
				# losses.update(loss.data.item(), inputs.size(0))
				# top1.update(prec1.item(), inputs.size(0))
				# top5.update(prec5.item(), inputs.size(0))
				
				# compute gradient and do SGD step
				self.classifier_optimizer.zero_grad()
				loss.backward()
				self.classifier_optimizer.step()
				new_weights.append(deepcopy(self.net.state_dict()))
			self.net.load_state_dict({name: weights_original[name] for name in weights_original})
		
		ws = len(new_weights)
		fweights = {name: new_weights[0][name] / float(ws) for name in new_weights[0]}
		for i in range(1, ws):
			# cur_weights = deepcopy(model.state_dict())
			for name in new_weights[i]:
				fweights[name] += new_weights[i][name] / float(ws)
		
		self.net.load_state_dict({name: weights_original[name] + ((fweights[name] - weights_original[name]) * meta_step_size) for name in weights_original})
		
		# Save model parameters
		if self.meta_step_count % Config.train.save_checkpoints_steps == 0:
			utils.save_checkpoint(self.meta_step_count, self.model_path, self.net, self.classifier_optimizer)
		
		return  # losses.avg, top1.avg
	
	def evaluate_model(self, dataset, mode="total_test"):
		self.net.train()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()
		old_model_state, test_set, train_set = self.learn_for_eval(dataset)
		num_correct, len_set = self._test_predictions(train_set, test_set)  # testing on only 1 sample mabye redundant
		self.net.load_state_dict(old_model_state)  # load back model's weights
		
		# prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
		# losses.update(loss.data.item(), inputs.size(0))
		# top1.update(prec1.item(), inputs.size(0))
		# top5.update(prec5.item(), inputs.size(0))
		# Step Verbose & Tensorboard Summary
		# if step_count % Config.train.verbose_step_count == 0:
		# 	self._add_summary(step_count, {f"loss_{mode}": losses.avg})
		# 	self._add_summary(step_count, {f"top1_acc_{mode}": top1.avg})
		# 	self._add_summary(step_count, {f"top5_acc_{mode}": top5.avg})
		# num_correct = float(sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)]))
		# self.prev_meta_step_count = step_count
		if mode == "total_test":
			return num_correct, len_set
		return num_correct
	
	def learn_for_eval(self, dataset):
		train_set, test_set = _split_train_test(_sample_mini_dataset(dataset, self.num_classes, self.num_shots + 1))  # 1 more sample for train
		model_state = deepcopy(self.net.state_dict())  # store weights to avoid training
		mini_batches = _mini_batches(train_set, Config.eval.eval_inner_iters, Config.eval.eval_inner_iters, self.replacement)
		# train on mini batches of the test set
		for batch_idx, batch in enumerate(mini_batches):
			inputs, labels = zip(*batch)
			inputs = Variable(torch.stack(inputs))
			labels = Variable(torch.from_numpy(np.array(labels)))
			if self.use_cuda:
				inputs = inputs.cuda()
				labels = labels.cuda()
			
			# compute output
			outputs = self.net(inputs)
			loss = self.loss_criterion(outputs, labels)  # measure accuracy and record loss
			self.classifier_optimizer.zero_grad()
			loss.backward()
			self.classifier_optimizer.step()
		return model_state, test_set, train_set
	
	def predict(self, criterion):
		print("Predicting on test set...")
		if self.use_cuda:
			self.net.cuda()
		if self.classifier_optimizer is None:
			self.classifier_optimizer = self.build_optimizers(self.net)
		self.loss_criterion = criterion
		
		# Load model
		self.prev_meta_step_count, self.net, self.classifier_optimizer = utils.load_saved_model(self.model_path, self.net, self.classifier_optimizer)
		print(f"Model has been loaded step:{self.prev_meta_step_count}, path:{self.model_path}")
		
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
		self.net.eval()
		num_correct = 0
		test_inputs, test_labels = zip(*test_set)
		if self._transductive:
			if self.use_cuda:
				test_inputs = Variable(torch.stack(test_inputs)).cuda()
				num_correct += sum(np.argmax(self.net(test_inputs).cpu().detach().numpy(), axis=1) == test_labels)
			else:
				test_inputs = Variable(torch.stack(test_inputs))
				num_correct += sum(np.argmax(self.net(test_inputs).cpu().detach().numpy(), axis=1) == test_labels)
			return num_correct, len(test_labels)
		res = []
		for test_sample in test_set:
			train_inputs, train_labels = zip(*train_set)
			if self.use_cuda:
				train_inputs = Variable(torch.stack(train_inputs)).cuda()
				train_inputs += Variable(torch.stack((test_sample[0],))).cuda()
			else:
				train_inputs = Variable(torch.stack(train_inputs))
				train_inputs += Variable(torch.stack((test_sample[0],)))
			argmax_arr = np.argmax(self.net(train_inputs).cpu().detach().numpy(), axis=1)
			res.append(argmax_arr[-1])
		num_correct += count_correct(res, test_labels)
		# res.append(np.argmax(self.net(inputs).cpu().detach().numpy(), axis=1))
		return num_correct, len(res)
	
	def evaluate(self, dataset, num_classes=5, num_samples=10000):
		"""
		Evaluate a model on a dataset. Final test!
		"""
		acc_all = []
		for i in range(num_samples):
			correct_this, count_this = self.evaluate_model(dataset, mode="total_test")
			acc_all.append(correct_this / count_this * 100)
			# print(f"eval: step:{i}, current_currect:{correct_this}, total_query:{count_this}")
			if i % 50 == 5:
				acc_arr = np.asarray(acc_all)
				acc_mean = np.mean(acc_arr)
				acc_std = np.std(acc_arr)
				print('Step:%d | Test Acc:%4.2f%% +-%4.2f%%' % (i, acc_mean, 1.96 * acc_std / np.sqrt(i)))
		
		return acc_mean


def count_correct(pred, target):
	''' count number of correct classification predictions in a batch '''
	pairs = [int(x == y) for (x, y) in zip(pred, target)]
	return sum(pairs)
