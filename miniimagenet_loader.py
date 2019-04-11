"""
Loading and using the Mini-ImageNet dataset.
To use these APIs, you should prepare a directory that
contains three sub-directories: train, test, and val.
Each of these three directories should contain one
sub-directory per WordNet ID.
"""

import os
import random

from PIL import Image, ImageFile
from hbconfig import Config
from torchvision.transforms import transforms, ToTensor

from AutoAugment.autoaugment import ImageNetPolicy


ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_dataset_test(data_dir, transforms=None):
	"""
	Read the Mini-ImageNet dataset.
	Args:
	  data_dir: directory containing Mini-ImageNet.
	Returns:
	  A tuple (train, val, test) of sequences of
		ImageNetClass instances.
	"""
	return tuple([_read_classes(os.path.join(data_dir, 'test'), transforms),_read_classes(os.path.join(data_dir, 'test'), None)])  # , 'test'



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

			cur_batch.append(totensor(sample[0]))
			if len(cur_batch) < batch_size:
				continue
			yield cur_batch
			cur_batch = []
			batch_count += 1
			if batch_count == num_batches:
				return

def _mini_batches_with_augmentation(samples, batch_size, num_batches, replacement,num_aug=5):
	policy = ImageNetPolicy()
	totensor = ToTensor()
	samples = list(samples)
	if replacement:
		for _ in range(num_batches):
				for x in  samples:
					samples=[(totensor(policy(x[0])),x[1]) for _ in range(num_aug)]
				yield random.sample(samples, batch_size)
		return
	cur_batch = []
	batch_count = 0
	while True:
		random.shuffle(samples)
		for sample in samples:
			if isinstance(sample[0], list):
				sample = (sample[0],sample[1])
			new_samples=[(totensor(policy(sample[0])[0]),sample[1]) for _ in range(num_aug)]
			cur_batch.extend(new_samples)
			if len(cur_batch)//num_aug < batch_size:
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

#TODO 1. check test_predictin method
#     2. update meta-optimizer