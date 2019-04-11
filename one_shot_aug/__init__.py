import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from hbconfig import Config
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

from AutoAugment.autoaugment import ImageNetPolicy
from logger import Logger
from miniimagenet_loader import read_dataset_test, _sample_mini_dataset, _mini_batches, _split_train_test
from one_shot_aug.module import PretrainedClassifier, MiniImageNetModel
from one_shot_aug.utils import AverageMeter, accuracy, mkdir_p
from utils import saving_config
from . import utils


meta_step_size = 1.  # stepsize of outer optimization, i.e., meta-optimization
meta_step_size_final = 0.


def augments_dataset(batch, k =5):
	import matplotlib.pyplot as plt
	
	images= []
	# labels=[]
	for _ in range(k):
		for img_,label in batch:
			policy = ImageNetPolicy()
			transformed = policy(img_)
			# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
			# ax1.imshow(img_)
			# ax2.imshow(transformed[0])
			# plt.show()
			tensor=ToTensor()
			if isinstance(transformed, (list,)):
				images.append((tensor(transformed[0]),label))
		else:
			images.append((tensor(transformed),label))
			# labels.append(label)
		images.append((tensor(img_),label))
	return images

class OneShotAug():
	model_name = "OneShotAug"
	
	# os.makedirs('images', exist_ok=True)
	
	def __init__(self):
		self._transductive = Config.model.transductive
		self.use_cuda = True if torch.cuda.is_available() else False
		self.device = torch.device("cuda" if self.use_cuda else "cpu")
		# self.tensorboard = utils.TensorBoard(Config.train.model_dir)
		# self.classifier_optimizer = None
		self.num_classes = Config.model.n_classes
		self.train_shot = Config.model.train_shot
		self.num_shots = Config.model.num_smaples_in_shot
		self.inner_batch_size = Config.train.inner_batch_size
		self.inner_iters = Config.train.inner_iters
		self.replacement = Config.train.replacement
		self.meta_batch_size = Config.train.meta_batch_size
		# load model
		# self.classifier.model.load_state_dict(torch.load(os.path.join(self.model_path + str(Config.model.name) + '.t7')))
		if Config.model.type == "known_net":
			self.meta_net = PretrainedClassifier()
			self.meta_net_base = self.meta_net
			self.title = self.meta_net.title
			self.meta_net = self.meta_net.model
		else:
			self.meta_net = MiniImageNetModel()
			self.title = self.meta_net.title
		self.learning_rate = Config.train.learning_rate
		self.c_path = f"{Config.train.model_dir}/log"
		self.model_path = f"{Config.train.model_dir}/model"
		mkdir_p(self.c_path)
		mkdir_p(self.model_path)
		# Print Config setting
		saving_config(os.path.join(self.c_path, "config_log.txt"))
	
	def train_fn(self, criterion, optimizer, resume=True):
		self.loss_criterion = criterion
		self.fast_optimizer = optimizer
		self.meta_optimizer = torch.optim.SGD(self.meta_net.parameters(), lr=Config.train.meta_lr)
		
		# self.exp_lr_scheduler = lr_scheduler.StepLR(self.classifier_optimizer, step_size=10, gamma=0.1)
		
		if resume:
			self.prev_meta_step_count, self.meta_net, self.meta_optimizer, self.state = utils.load_saved_model(self.model_path, self.meta_net, self.meta_optimizer)
			print(f"Model has been loaded step:{self.prev_meta_step_count}, path:{self.model_path}")
		self.logger = Logger(os.path.join(self.c_path, 'log.txt'), title=self.title)
		self.logger.set_names(['step', 'Learning Rate', 'Train Acc.', 'Valid Acc.'])
		return self._train
	
	def _train(self, data_loader):
		self.meta_net.train()
		train_loader, validation_loader = data_loader
		# switch to train mode
		# if self.classifier.arch.startswith('alexnet') or self.classifier.arch.startswith('vgg'):
		if self.use_cuda:
			self.meta_net.cuda()
			# self.net = torch.nn.DataParallel(self.net).to(self.device)
			# if torch.cuda.device_count() > 1:
			# 	print("Using ", torch.cuda.device_count(), " GPUs!")
			# cudnn.benchmark = True
		print('Total params: %.2fK' % (sum(p.numel() for p in self.meta_net.parameters()) / 1000.0))
		for current_meta_step in range(self.prev_meta_step_count, Config.train.meta_iters):
			fast_net = self.meta_net.clone(self.use_cuda)
			optimizer = get_optimizer(fast_net, self.state)
			# Training
			self._train_step(fast_net, train_loader, current_meta_step, optimizer)
			self.state = optimizer.state_dict()  # save optimizer state
			
			# Evaluation
			if current_meta_step % Config.train.verbose_step_count == 0:
				fast_net = self.meta_net.clone(self.use_cuda)
				optimizer = get_optimizer(fast_net, self.state)
				validation_num_correct, valid_count = self.evaluate_model(fast_net, optimizer, validation_loader)
				# self._add_summary(current_meta_step, {"loss_valid": validation_loss})
				# self._add_summary(current_meta_step, {"top1_acc_valid": validation_acc})
				valid_acc_eval = float(validation_num_correct) / valid_count
				# self._add_summary(current_meta_step, {"accuracy_valid": valid_acc_eval})
				train_num_correct, train_count = self.evaluate_model(fast_net, optimizer,train_loader)
				# self._add_summary(current_meta_step, {"loss_train": train_loss})
				# self._add_summary(current_meta_step, {"top1_acc_train": train_acc})
				train_acc_eval = float(train_num_correct) / train_count
				# self._add_summary(current_meta_step, {"accuracy_train": train_acc_eval})
				print(f"step{current_meta_step}| accuracy_train: {train_acc_eval}| accuracy_valid:{valid_acc_eval}")
				# loading back optimizer state
				# optimizer.load_state_dict(state)
				self.logger.append([current_meta_step, self.learning_rate, train_acc_eval, valid_acc_eval])
				
				# TODO: implement update when lowest loss
				# if validation_loss < best_loss:
				# 	best_loss = validation_loss
				# 	best_model_wts = deepcopy(self.classifier.state_dict())
				
				# update learning rate
				# self.exp_lr_scheduler.step()
		
		# predict on test set after training finished
		self.predict(self.loss_criterion)
	
	def _train_step(self,fast_net, train_loader, current_meta_step,optimizer):
		# weights_original = self.meta_net.clone(self.use_cuda)#parameters_to_vector(self.net.parameters().clone())
		new_weights = []
		for _ in range(self.meta_batch_size):
			new_weights.append(self.inner_train(fast_net, train_loader,optimizer))
			# vector_to_parameters(weights_original, self.net.parameters())
		# self.net.load_state_dict(weights_original.state_dict())
		self.interpolate_new_weights(new_weights, fast_net, current_meta_step)
		self.meta_optimizer.step()
		
		# Save model parameters
		if current_meta_step > 0 and current_meta_step  % Config.train.save_checkpoints_steps == 0:
			utils.save_checkpoint(current_meta_step, self.model_path, self.meta_net, self.meta_optimizer,state=optimizer.state_dict())
		return  # losses.avg, top1.avg
	
	def interpolate_new_weights(self, new_weights, net, current_meta_step):
		# weights_original=parameters_to_vector(weights_original)
		frac_done = current_meta_step / Config.train.meta_iters
		cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
		
		self.average_weights(new_weights, net) # TODO verify this is an average weights
		self.meta_net.point_grad_to(net, self.use_cuda, cur_meta_step_size)
		
		# vector_to_parameters(weights_original + (fweights-weights_original)* cur_meta_step_size, self.meta_net.parameters())
		# b = list(self.net.parameters())[-1].clone()
		# print(f"IS EQUAL {torch.equal(a.data, b.data)}")
		# self.net.load_state_dict({name: weights_original[name] + ((fweights[name] - weights_original[name]) * cur_meta_step_size) for name in weights_original})
	
	def average_weights(self, params_list, net):
		avg_param = deepcopy(list(1/float(len(params_list)) * p.data for p in net.parameters()))
		
		#zero grads
		for avg_p in avg_param:
			if avg_p.grad is None:
				if self.use_cuda:
					avg_p.grad = Variable(torch.zeros(avg_p.size())).cuda()
				else:
					avg_p.grad = Variable(torch.zeros(avg_p.size()))
				avg_p.grad.data.zero_()  # not sure this is required
		#averaging
		for i in range(1,len(params_list)):
			for avg_p, target_p in zip(avg_param, params_list[i]):
				avg_p.add_(1/float(len(params_list)) * target_p.data)
		# load to model
		for p, avg_p in zip(net.parameters(), avg_param):
			p.data.copy_(avg_p)
		
		
		
		# num_weights = len(new_weights)
		# fweights = {name: new_weights[0][name] / float(num_weights) for name in new_weights[0]}
		# for i in range(1, num_weights):
		# 	for name in new_weights[i]:
		# 		fweights[name] += new_weights[i][name] / float(num_weights)
		# return fweights
	
	def inner_train(self, fast_net, train_loader, optimizer):
		fast_net.train()
		mini_data_set = _sample_mini_dataset(train_loader, self.num_classes, self.train_shot)
		mini_train_loader = _mini_batches(mini_data_set, self.inner_batch_size, self.inner_iters, self.replacement)
		for batch_idx, batch in enumerate(mini_train_loader):
			# init value
			inputs, labels = zip(*batch)
			# show_images(inputs,labels)
			inputs = Variable(torch.stack(inputs))
			labels = Variable(torch.from_numpy(np.array(labels)))
			if self.use_cuda:
				inputs = inputs.cuda()
				labels = labels.cuda()
			
			# compute output
			outputs = fast_net(inputs)
			loss = self.loss_criterion(outputs, labels)
			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
		return deepcopy(list(p.data for p in fast_net.parameters()))
	
	def evaluate_model(self, fast_net, optimaizer, dataset, mode="total_test"):
		# old_model_state = deepcopy(fast_net.state_dict())  # store weights to avoid training
		train_set_imgs, _ = _split_train_test(_sample_mini_dataset(dataset[0], self.num_classes, self.num_shots + 1))  # 1 more sample for train
		train_set_tensors, test_set_tensors = _split_train_test(_sample_mini_dataset(dataset[1], self.num_classes, self.num_shots + 1))  # 1 more sample for train
		self.learn_for_eval(fast_net, optimaizer,train_set_imgs)
		num_correct, len_set = self._test_predictions(fast_net, train_set_tensors, test_set_tensors)  # testing on only 1 sample mabye redundant
		
		# self.net.load_state_dict(old_model_state)  # load back model's weights
		
		if mode == "total_test":
			return num_correct, len_set
		return num_correct
	
	def learn_for_eval(self, fast_net, optimaizer, train_set):
		fast_net.train()
		mini_batches = _mini_batches(train_set, Config.eval.inner_batch_size, Config.eval.eval_inner_iters, self.replacement)
		# train on mini batches of the test set
		for batch_idx, batch in enumerate(mini_batches):
			augmented_dataset = augments_dataset(batch)
			inputs, labels =  zip(*augmented_dataset)
			# show_images(inputs, labels)
			inputs = Variable(torch.stack(inputs))
			labels = Variable(torch.from_numpy(np.array(labels)))
			if self.use_cuda:
				inputs = inputs.cuda()
				labels = labels.cuda()
			# compute output
			outputs = fast_net(inputs)
			loss = self.loss_criterion(outputs, labels)  # measure accuracy and record loss
			optimaizer.zero_grad()
			loss.backward()
			optimaizer.step()

		
		return
	
	def predict(self, criterion):
		print("Predicting on test set...")
		if self.use_cuda:
			self.meta_net.cuda()
		self.loss_criterion = criterion
		
		# Load model
		self.prev_meta_step_count, self.meta_net, self.meta_optimizer, self.state = utils.load_saved_model(self.model_path, self.meta_net, self.build_optimizers(self.meta_net))
		print(f"Model has been loaded step:{self.prev_meta_step_count}, path:{self.model_path}")
		transform_list_test = []
		# if Config.predict.use_augmentation:
		# transform_list_test.extend([transforms.Resize(Config.data.image_size), ImageNetPolicy(Config.predict.num_sample_augmentation)])
		transform_list_test.extend([transforms.Resize(Config.data.image_size),
		                            transforms.ToTensor(),
		                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
		                                                 std=[0.229, 0.224, 0.225])
		                            ])
		
		transform_test = transforms.Compose(transform_list_test)
		test_dataset_tesnors,test_dataset_imgs = read_dataset_test(Config.data.miniimagenet_path,transform_test)
		evaluation = self.evaluate(test_dataset_imgs,test_dataset_tesnors)
		print(f"Total score: {evaluation}")
		return evaluation
	
	def build_criterion(self):
		return torch.nn.CrossEntropyLoss().to(self.device)
	
	def build_optimizers(self, classifier):
		classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=self.learning_rate*2, betas=(0,0.999))
		
		return classifier_optimizer
	
	def _add_summary(self, step, summary):
		for tag, value in summary.items():
			self.tensorboard.scalar_summary(tag, value, step)
	
	def _test_predictions(self,fast_net, train_set, test_set):
		fast_net.eval()
		num_correct = 0
		test_inputs, test_labels = zip(*test_set)
		
		if self._transductive:
			if self.use_cuda:
				test_inputs = Variable(torch.stack(test_inputs)).cuda()
			else:
				test_inputs = Variable(torch.stack(test_inputs))
			num_correct += sum(np.argmax(fast_net(test_inputs).cpu().detach().numpy(), axis=1) == test_labels)
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
			argmax_arr = np.argmax(fast_net(train_inputs).cpu().detach().numpy(), axis=1)
			res.append(argmax_arr[-1])
		num_correct += count_correct(res, test_labels)
		# res.append(np.argmax(self.net(inputs).cpu().detach().numpy(), axis=1))
		return num_correct, len(res)
	
	def evaluate(self, dataset_tesnsors,dataset_imgs, num_samples=10000):
		"""
		Evaluate a model on a dataset. Final test!
		"""
		acc_all = []
		for i in range(num_samples):
			
			fast_net = deepcopy(self.meta_net)
			optimizer = get_optimizer(fast_net,self.state)
			
			correct_this, count_this = self.evaluate_model(fast_net, optimizer, (dataset_tesnsors,dataset_imgs), mode="total_test")
			acc_all.append(correct_this / count_this * 100)
			# print(f"eval: step:{i}, current_currect:{correct_this}, total_query:{count_this}")
			if i % 50 == 5:
				acc_arr = np.asarray(acc_all)
				acc_mean = np.mean(acc_arr)
				acc_std = np.std(acc_arr)
				print('Step:%d | Test Acc:%4.2f%% +-%4.2f%%' % (i, acc_mean, 1.96 * acc_std / np.sqrt(i)))
		
		return acc_mean
	
def get_optimizer(net, state=None):
	optimizer = torch.optim.Adam(net.parameters(), lr=Config.train.learning_rate, betas=(0, 0.999))
	if state is not None:
		optimizer.load_state_dict(state)
	return optimizer

def count_correct(pred, target):
	''' count number of correct classification predictions in a batch '''
	pairs = [int(x == y) for (x, y) in zip(pred, target)]
	return sum(pairs)


# Show Image
def show_image(image):
	import matplotlib.pyplot as plt
	# Convert image to numpy
	image = image.numpy()
	
	# Un-normalize the image
	# image[0] = image[0] * [0.229, 0.224, 0.225] +[0.485, 0.456, 0.406]
	
	# Print the image
	plt.imshow(np.transpose(image, (1, 2, 0)), interpolation='nearest')
	# plt.imshow(np.transpose(image, (1, 2, 0)))
	plt.show()# Show Image

def show_images(images,labels):
	import matplotlib.pyplot as plt
	# Convert image to numpy
	import matplotlib.pyplot as plt
	fig=plt.figure(figsize=(10, 10))
	columns = 4
	rows = 5
	for i in range(1, columns*rows+1):
		ax = fig.add_subplot(rows, columns, i)
		ax.set_title(labels[i])
		image = images[i].numpy()
		plt.imshow(np.transpose(image, (1, 2, 0)), interpolation='nearest')
	plt.show()

def set_learning_rate(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr