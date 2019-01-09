"""
Loading and using the Mini-ImageNet dataset.
To use these APIs, you should prepare a directory that
contains three sub-directories: train, test, and val.
Each of these three directories should contain one
sub-directory per WordNet ID.
"""

import os
import random

from PIL import Image
from hbconfig import Config
from torchvision.transforms import transforms


def read_dataset_test(data_dir, transforms=None):
	"""
	Read the Mini-ImageNet dataset.
	Args:
	  data_dir: directory containing Mini-ImageNet.
	Returns:
	  A tuple (train, val, test) of sequences of
		ImageNetClass instances.
	"""
	return tuple(_read_classes(os.path.join(data_dir, x), transforms) for x in ['test'])


def read_dataset(data_dir, transforms=None):
	"""
	Read the Mini-ImageNet dataset.
	Args:
	  data_dir: directory containing Mini-ImageNet.
	Returns:
	  A tuple (train, val, test) of sequences of
		ImageNetClass instances.
	"""
	return tuple(_read_classes(os.path.join(data_dir, x), transforms) for x in ['train', 'val'])  # , 'test'


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
		if transform is None:
			self.transform = transforms.Compose(
				[transforms.RandomResizedCrop(Config.data.image_size),
				 transforms.RandomHorizontalFlip(),
				 transforms.ToTensor(),
				 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
		else:
			self.transform = transform
	
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


def inner_mini_batches(self, mini_dataset, inner_batch_size, inner_iters, replacement):
	"""
	Generate inner-loop mini-batches for the task.
	"""
	if self.tail_shots is None:
		for value in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
			yield value
		return
	train, tail = _split_train_test(mini_dataset, test_shots=self.tail_shots)
	for batch in _mini_batches(train, inner_batch_size, inner_iters - 1, replacement):
		yield batch
	yield tail


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
