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
from torch.optim import Adam
from torchsummary import summary
from torchvision.transforms import transforms, ToTensor

from AutoAugment.autoaugment import ImageNetPolicy

from auto_encoder import *
from utils import find_latest, mkdir_p, get_sorted_path, _conv_layer, _conv_transpose_layer


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



def read_dataset(data_dir, transform_train=None,transform_test=None):
	"""
	Read the Mini-ImageNet dataset.
	Args:
	  data_dir: directory containing Mini-ImageNet.
	Returns:
	  A tuple (train, val, test) of sequences of
		ImageNetClass instances.
	"""
	return tuple([_read_classes(os.path.join(data_dir, 'train'), transform_train),_read_classes(os.path.join(data_dir, 'val'), transform_test)])  # , 'test'


def _read_classes(dir_path, transforms):
	"""
	Read the WNID directories in a directory.
	"""
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
			self.transform = transforms.Compose(
				[transforms.Resize(Config.data.image_size),
				 # transforms.ToTensor(),
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
			img = Image.open(in_file).resize((84, 84)).convert('RGB')
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
	totensor= ToTensor()
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
			cur_batch.append((totensor(sample[0]),sample[1]))
			if len(cur_batch) < batch_size:
				continue
			yield cur_batch
			cur_batch = []
			batch_count += 1
			if batch_count == num_batches:
				return


class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()
		self.path_to_save = f"ae/model"
		# conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
		self.n_filters = 32
		self.layer1 = _conv_layer(3, self.n_filters, 3, 1, 0)
		self.layer2 = _conv_layer(self.n_filters, self.n_filters // 2, 3, 1,0)
		self.layer3 = _conv_layer(self.n_filters // 2, self.n_filters // 4, 3, 1,0)
		self.emb = nn.Sequential(nn.Linear(512, 512))
		
		# deconv layers: (in_channel size, out_channel size, kernel_size, stride, padding, output_padding)
		self.deconv1 = _conv_transpose_layer(self.n_filters // 4, self.n_filters // 2, 3, stride=3, padding=1, output_padding=1)
		self.deconv2 = _conv_transpose_layer(self.n_filters // 2, self.n_filters, 3, stride=2, padding=2, output_padding=1)
		self.deconv3 = _conv_transpose_layer(self.n_filters, 3, 3, stride=2, padding=3, output_padding=1)
		mkdir_p(self.path_to_save)
		summary(self, (3, 84, 84))
	
	def forward(self, x):
		# the autoencoder has 3 con layers and 3 deconv layers (transposed conv). All layers but the last have ReLu
		# activation function
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = x.view(x.size()[0], -1)
		x = self.emb(x)
		x = x.reshape(-1, 8, 8, 8)
		x = self.deconv1(x)
		x = self.deconv2(x)
		self.decoded = self.deconv3(x)
		x = torch.tanh(self.decoded)
		return x.reshape((-1,3,84,84))
	
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
			model.optimizer.load_state_dict(checkpoint['optimizer'])
		except:
			# else:
			model.load_state_dict(checkpoint['net'])
			if model.optimizer is not None:
				model.optimizer.load_state_dict(checkpoint['optimizer'])
		
		print(f"Load checkpoints...! {latest_path}")
		return step_count, model
	
	def set_optimizer(self):
		self.optimizer = Adam(self.parameters(), lr=1e-3)
	
	def save_checkpoint(self, step, max_to_keep=3):
		sorted_path = get_sorted_path(self.path_to_save)
		for i in range(len(sorted_path) - max_to_keep):
			os.remove(sorted_path[i])
		
		full_path = os.path.join(self.path_to_save, f"ae_{step}.pkl")
		torch.save({"step_count": step, 'net': self.state_dict(), 'optimizer': self.optimizer.state_dict(), }, full_path)
		print(f"Save checkpoints...! {full_path}")


def _mini_batches_with_augmentation(samples, batch_size, num_batches, replacement,num_aug=5):
	ae= AutoEncoder()
	epoch, ae = ae.load_saved_model(ae.path_to_save, ae)
	ae.eval()
	print(f"AutoEncoer has been loaded epoch:{epoch}, path:{ae.path_to_save}")
	
	policy =  ae # ImageNetPolicy()
	samples = list(samples)
	cur_batch = []
	if replacement:
		for _ in range(num_batches):
			for _ in range(num_aug):
				for x in  samples:
					cur_batch.append((totensor(policy(x[0])),x[1]))
			yield random.sample(cur_batch, batch_size)
		return
	batch_count = 0
	while True:
		random.shuffle(samples)
		for idx in range(num_aug):
			for sample in samples:
				if idx == 0:
					cur_batch.append((totensor(sample[0]),sample[1]))
				else:
					cur_batch.append((policy(torch.unsqueeze(totensor(sample[0]),0)).squeeze(),sample[1]))
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
