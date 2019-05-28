import numpy as np
import torch
from hbconfig import Config
from sklearn.datasets import make_moons
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms

from AutoAugment.autoaugment import ImageNetPolicy
from miniimagenet_loader import read_dataset


def get_loader(mode):
	"""Builds and returns Dataloader for MNIST and SVHN dataset."""
	global train_loader, valid_loader
	config = Config
	transform_list_train = []
	transform_list_test = []
	is_train = mode == "train"
	if config.train.use_augmentation:
		transform_list_train.extend([transforms.Resize((config.data.image_size, config.data.image_size)), ImageNetPolicy()])
	transform_list_train.extend([transforms.Resize((config.data.image_size, config.data.image_size)), transforms.ToTensor(),
	                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	
	transform_train = transforms.Compose(transform_list_train)
	
	if config.predict.use_augmentation:
		transform_list_test.extend([transforms.Resize((config.data.image_size, config.data.image_size)), ImageNetPolicy()])
	transform_list_test.extend([transforms.Resize((config.data.image_size, config.data.image_size)), transforms.ToTensor(),
	                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	
	transform_test = transforms.Compose(transform_list_test)
	if config.model.dataset == "mnist":
		mnist = datasets.MNIST(root=config.data.mnist_path, download=True, transform=transform_train, train=is_train)
		# train-validation split
		train_mnist, valid_mnist = train_valid_split(mnist)
		train_loader = DataLoader(dataset=train_mnist, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
		valid_loader = DataLoader(dataset=valid_mnist, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
	if config.model.dataset == "svhn":
		svhn = datasets.SVHN(root=config.data.svhn_path, download=True, transform=transform_train, split=mode)
		train_svhn, valid_svhn = train_valid_split(svhn)
		train_loader = DataLoader(dataset=train_svhn, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
		valid_loader = DataLoader(dataset=valid_svhn, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
	if config.model.dataset == "cifar10":
		cifar10 = datasets.CIFAR10(root=config.data.cifar10_path, download=True, transform=transform_train, train=is_train)
		train_cifar, valid_cifar = train_valid_split(cifar10)
		train_loader = DataLoader(dataset=train_cifar, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
		valid_loader = DataLoader(dataset=valid_cifar, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
	if config.model.dataset == "moons":
		train_moons, valid_moons = train_valid_split(moons_dataset())
		train_loader = DataLoader(dataset=train_moons, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
		valid_loader = DataLoader(dataset=valid_moons, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
	if config.model.dataset == "miniimagenet":
		train_loader, valid_loader = read_dataset(Config.data.miniimagenet_path, transform_train, transform_test)  # transform_train(train_loader)
	if config.model.dataset == "miniimagenet_all":
		train_loader = datasets.ImageFolder(root=config.data.miniimagenet_path_train, transform=transform_train)
		valid_loader = datasets.ImageFolder(root=config.data.miniimagenet_path_valid, transform=transform_list_test)
		# 	# test_imagenet = datasets.ImageFolder(root=config.data.miniimagenet_path_test)
		train_loader = DataLoader(dataset=train_loader, batch_size=Config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
		valid_loader = DataLoader(dataset=valid_loader, batch_size=Config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
	# 	# test_loader = DataLoader(dataset=test_imagenet, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
	if config.model.dataset == "miniimagenet_concat":
		concat_dataset = ConcatDataset([datasets.ImageFolder(config.data.miniimagenet_path_train, transform=transform_train),
		                        datasets.ImageFolder(config.data.miniimagenet_path_valid, transform=transform_train)])
		train_, valid_ = train_valid_split(concat_dataset)
		train_loader = torch.utils.data.DataLoader(train_, batch_size=config.train.batch_size, shuffle=True, num_workers=config.data.num_workers, pin_memory=True)
		valid_loader = torch.utils.data.DataLoader(valid_, batch_size=config.train.batch_size, shuffle=True, num_workers=config.data.num_workers, pin_memory=True)
	
	if config.model.dataset == "imagenet":
		train_loader = datasets.ImageFolder(root=config.data.imagenet_path_train, transform=transform_train)
		valid_loader = datasets.ImageFolder(root=config.data.imagenet_path_val, transform=transform_list_test)
		# 	# test_imagenet = datasets.ImageFolder(root=config.data.miniimagenet_path_test)
		train_loader = DataLoader(dataset=train_loader, batch_size=Config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
		valid_loader = DataLoader(dataset=valid_loader, batch_size=Config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
	return train_loader, valid_loader




"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""


class GenHelper(Dataset):
	def __init__(self, mother, length, mapping):
		# here is a mapping from this index to the mother ds index
		self.mapping = mapping
		self.length = length
		self.mother = mother
	
	def __getitem__(self, index):
		return self.mother[self.mapping[index]]
	
	def __len__(self):
		return self.length


def train_valid_split(dataset, split_fold=10, random_seed=None):
	'''
	This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
	efficiently.
	:return:
	'''
	if random_seed != None:
		np.random.seed(random_seed)
	
	dslen = len(dataset)
	indices = list(range(dslen))
	valid_size = dslen // split_fold
	np.random.shuffle(indices)
	train_mapping = indices[valid_size:]
	valid_mapping = indices[:valid_size]
	train = GenHelper(dataset, dslen - valid_size, train_mapping)
	valid = GenHelper(dataset, valid_size, valid_mapping)
	
	return train, valid


def to_categorical(y, num_classes):
	"""1-hot encodes a tensor"""
	return np.eye(num_classes, dtype='uint8')[y]


class PrepareData(Dataset):
	
	def __init__(self, X, y):
		if not torch.is_tensor(X):
			self.X = torch.from_numpy(X)
		if not torch.is_tensor(y):
			self.y = torch.from_numpy(y)
	
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


def moons_dataset():
	X, y = make_moons(n_samples=1000, noise=.1)
	y = to_categorical(y, 2)
	ds = PrepareData(X=X, y=y)
	return ds  # ds = DataLoader(ds, batch_size=50, shuffle=True)
