"""
Loading and using the Mini-ImageNet dataset.
To use these APIs, you should prepare a directory that
contains three sub-directories: train, test, and val.
Each of these three directories should contain one
sub-directory per WordNet ID.
"""

import os
import random
from collections import OrderedDict

import torch
from PIL import Image, ImageFile
from hbconfig import Config
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.transforms import transforms, ToTensor

from AutoAugment.autoaugment import ImageNetPolicy
from utils import find_latest, mkdir_p, get_sorted_path


ImageFile.LOAD_TRUNCATED_IMAGES = True
totensor = ToTensor()


def read_dataset_test(data_dir, transforms=None):
	"""
	Read the Mini-ImageNet dataset.
	Args:
	  data_dir: directory containing Mini-ImageNet.
	Returns:
	  A tuple (train, val, test) of sequences of
		ImageNetClass instances.
	"""
	return tuple([_read_classes(os.path.join(data_dir, 'test'), transforms)])


def read_dataset(data_dir, transform_train=None, transform_test=None):
	"""
	Read the Mini-ImageNet dataset.
	Args:
	  data_dir: directory containing Mini-ImageNet.
	Returns:
	  A tuple (train, val, test) of sequences of
		ImageNetClass instances.
	"""
	return tuple([_read_classes(os.path.join(data_dir, 'train'), transform_train), _read_classes(os.path.join(data_dir, 'val'), transform_test)])  # , 'test'


def _read_classes(dir_path, transforms):
	"""
	Read the WNID directories in a directory.
	"""
	print(f"=>Number of test classes = {len([ f for f in os.listdir(dir_path) if f.startswith('n')])}")
	return [ImageNetClass(os.path.join(dir_path, f), transforms) for f in os.listdir(dir_path) if f.startswith('n')]


# pylint: disable=R0903
class ImageNetClass:
	"""
	A single image class.
	"""
	
	def __init__(self, dir_path, transform):
		self.dir_path = dir_path
		self._cache = {}
		self.transform = transform
		if transform is None:
			self.transform = transforms.Compose([transforms.Resize((Config.data.image_size, Config.data.image_size)),  # transforms.ToTensor(),
			                                     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
			                                     ])
	
	def __len__(self):
		return len([f for f in os.listdir(self.dir_path) if f.endswith('.JPEG')])
	
	def sample(self, num_images):
		"""
		Sample images (as numpy arrays) from the class.
		Returns:
		  A sequence of 84x84x3 numpy arrays.
		  Each pixel ranges from 0 to 1.
		"""
		names = [f for f in os.listdir(self.dir_path) if f.endswith('.JPEG')]
		random.shuffle(names)
		images = []
		for name in names[:num_images]:
			images.append(self._read_image(name))
		return images
	
	def _read_image(self, name):
		if name in self._cache:
			tmp = self._cache[name]  # .astype('float32') / 0xff
			
			return tmp
		with open(os.path.join(self.dir_path, name), 'rb') as in_file:
			img = Image.open(in_file).resize((Config.data.image_size, Config.data.image_size)).convert('RGB')
			self._cache[name] = self.transform(img)
			return self._read_image(name)


def _sample_mini_dataset(dataset, num_classes, num_shots):
	"""
	Sample a few shot task from a dataset.
	Returns:
	  An iterable of (input, label) pairs.
	"""
	shuffled = list(dataset)
	random.shuffle(shuffled)
	for class_idx, class_obj in enumerate(shuffled[:num_classes]):
		# num_shots = min(num_shots,len(class_obj))
		for sample in class_obj.sample(num_shots):
			yield (sample, class_idx)


def _mini_batches(samples, batch_size, num_batches, replacement):
	"""
	Generate mini-batches from some data.
	Returns:
	  An iterable of sequences of (input, label) pairs,
		where each sequence is a mini-batch.
	"""
	totensor = ToTensor()
	samples = list(samples)
	if replacement:
		for _ in range(num_batches):
			yield random.sample(totensor(samples), batch_size)
		return
	cur_batch = []
	batch_count = 0
	while True:
		random.shuffle(samples)
		for sample in samples:
			cur_batch.append((totensor(sample[0]), sample[1]))
			if len(cur_batch) < batch_size:
				continue
			yield cur_batch
			cur_batch = []
			batch_count += 1
			if batch_count == num_batches:
				return


class Interpolate(nn.Module):
	def __init__(self, mode, scale_factor):
		super(Interpolate, self).__init__()
		self.interp = nn.functional.interpolate
		self.size = scale_factor
		self.mode = mode
	
	def forward(self, x):
		x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
		return x

class GaussianNoise(nn.Module):
	"""Gaussian noise regularizer.

	Args:
		sigma (float, optional): relative standard deviation used to generate the
			noise. Relative means that it will be multiplied by the magnitude of
			the value your are adding the noise to. This means that sigma can be
			the same regardless of the scale of the vector.
		is_relative_detach (bool, optional): whether to detach the variable before
			computing the scale of the noise. If `False` then the scale of the noise
			won't be seen as a constant but something to optimize: this will bias the
			network to generate vectors with smaller values.
	"""
	
	def __init__(self, sigma=0.1, is_relative_detach=True):
		super().__init__()
		self.sigma = sigma
		self.is_relative_detach = is_relative_detach
		# self.noise = torch.FloatTensor(0)
	
	def forward(self, x):
		if self.training and self.sigma != 0:
			scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
			self.noise = torch.zeros(x.size())
			sampled_noise = self.noise.normal_() * scale
			x = x + sampled_noise
		return x
class DynamicGNoise(nn.Module):
	def __init__(self, shape, std=0.05):
		super().__init__()
		self.noise = Variable(torch.zeros(shape,shape).cuda())
		self.std   = std
	
	def forward(self, x):
		if not self.training: return x
		self.noise.data.normal_(0, std=self.std)
		
		print(x.size(), self.noise.size())
		return x + self.noise

def gaussian(ins, is_training=True, mean=0.0, stddev=0.1):
	if is_training:
		noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
		return ins + noise
	return ins
	
class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()
		self.path_to_save = Config.train.model_dir
		self.use_cuda = torch.cuda.is_available()
		# conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
		# self.n_filters = 32
		# self.layer1 = _conv_layer(3, self.n_filters, 3, 1, 0)
		# self.layer2 = _conv_layer(self.n_filters, self.n_filters // 2, 3, 1, 0)
		# self.layer3 = _conv_layer(self.n_filters // 2, self.n_filters // 4, 3, 1, 0)
		# # self.emb = nn.Sequential(nn.Linear(512, 512))
		# self.emb = nn.Sequential(nn.Linear(5408, 5408))
		
		# # deconv layers: (in_channel size, out_channel size, kernel_size, stride, padding, output_padding)
		# self.deconv1 = _conv_transpose_layer(self.n_filters // 4, self.n_filters // 2, 3, stride=3, padding=1, output_padding=1)
		# self.deconv2 = _conv_transpose_layer(self.n_filters // 2, self.n_filters, 3, stride=2, padding=2, output_padding=1)
		# self.deconv3 = _conv_transpose_layer(self.n_filters, 3, 3, stride=2, padding=3, output_padding=1)
		
		self.encoder = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(True), nn.MaxPool2d(2),
		
		                             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(True), nn.MaxPool2d(2),
		
		                             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(True), nn.MaxPool2d(2),  # nn.Linear(64*3*3,400),nn.LeakyReLU(True),
		                             # nn.Linear(400,256),nn.LeakyReLU(True)
		
		                             # nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.LeakyReLU(True), nn.MaxPool2d(2),
		                             #
		                             # nn.Conv2d(8, 8, kernel_size=3, padding=1), nn.LeakyReLU(True), nn.MaxPool2d(2))
		                             )
		self.decoder = nn.Sequential(  # nn.Linear(256,400), nn.LeakyReLU(True),
			# nn.Linear(400,3*3*64), nn.LeakyReLU(True),
			# Interpolate(mode='bilinear', scale_factor=2),
			# GaussianNoise(sigma=0.1),
			# DynamicGNoise(),
			nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(True),
			
			# Interpolate(mode='bilinear', scale_factor=2),
			nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(True),
			
			# Interpolate(mode='bilinear', scale_factor=2),
			nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(True),
			
			# Interpolate(mode='bilinear', scale_factor=2),
			# nn.ConvTranspose2d(32, 64, kernel_size=3,stride=2), nn.LeakyReLU(True),
			
			# Interpolate(mode='bilinear', scale_factor=2),
			# nn.ConvTranspose2d(64, 3, kernel_size=3,stride=2,padding=1), nn.LeakyReLU(True),
			
			nn.Tanh())
		
		mkdir_p(self.path_to_save)
	
	# if self.use_cuda:
	# 	self = self.cuda()
	# 	summary(self.cuda(), (Config.data.channels, Config.data.image_size, Config.data.image_size))
	# else:
	# 	summary(self, (Config.data.channels, Config.data.image_size, Config.data.image_size))
	def forward(self, x):
		# print("Start Encode: ", x.shape)
		if Config.train.add_noise:
			x= gaussian(x,mean=Config.train.noise_mean,stddev=Config.train.noise_std)
		x = self.encoder(x)
		# x+= gaussian(0.2*x[0],mean=0.1,stddev=0.1)
		# print("Finished Encode: ", x.shape)
		x = self.decoder(x)
		# print("Finished Decode: ", x.shape)
		return x
	
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
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				name = k[7:]  # remove 'module.' of dataparallel
				new_state_dict[name] = v
			
			model.load_state_dict(new_state_dict)
		except:
			# else:
			model.load_state_dict(checkpoint['net'])
		
		print(f"Load checkpoints...! {latest_path}")
		return step_count, model
	
	def set_optimizer(self):
		self.optimizer = Adam(self.parameters(), lr=Config.train.learning_rate)
	
	def save_checkpoint(self, step, max_to_keep=3):
		sorted_path = get_sorted_path(self.path_to_save)
		for i in range(len(sorted_path) - max_to_keep):
			os.remove(sorted_path[i])
		
		full_path = os.path.join(self.path_to_save, f"ae_{step}.pkl")
		torch.save({"step_count": step, 'net': self.state_dict(), 'optimizer': self.optimizer.state_dict(), }, full_path)
		print(f"Save checkpoints...! {full_path}")


def _mini_batches_with_augmentation(samples, batch_size, num_batches, replacement, num_aug=5, policy=None, use_cuda=False):
	policy = policy  # ImageNetPolicy()
	if policy is None:
		policy = ImageNetPolicy()
	samples = list(samples)
	cur_batch = []
	if replacement:
		for _ in range(num_batches):
			for _ in range(num_aug):
				for x in samples:
					cur_batch.append((totensor(policy(x[0])), x[1]))
			yield random.sample(cur_batch, batch_size)
		return
	batch_count = 0
	while True:
		random.shuffle(samples)
		for idx in range(num_aug):
			for sample in samples:
				if idx == 0:
					cur_batch.append((totensor(sample[0]), sample[1]))
				else:
					if use_cuda:
						img = torch.unsqueeze(totensor(sample[0]), 0).cuda()
					else:
						img = torch.unsqueeze(totensor(sample[0]), 0)
					if isinstance(policy,ImageNetPolicy):
						cur_batch.append((totensor(policy(sample[0])).squeeze(), sample[1]))
					else:
						cur_batch.append((policy(img).squeeze(), sample[1]))
					
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
