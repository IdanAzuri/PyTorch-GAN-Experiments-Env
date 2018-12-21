import numpy as np
import torch
import torchvision
from hbconfig import Config
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms



def get_loader(mode):
	"""Builds and returns Dataloader for MNIST and SVHN dataset."""
	global train_loader, valid_loader
	config = Config
	transform_list = []
	is_train = mode == "train"
	
	if config.model.use_augmentation:
		transform_list.extend[torchvision.transforms.Resize((224,224)),
		torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(p=0.2),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.RandomAffine(45),
		torchvision.transforms.RandomRotation(20),]
transform_list.extend([transforms.Resize(config.data.image_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		transform = transforms.Compose(transform_list)
	transforms = torchvision.transforms.Compose([
		torchvision.transforms.Resize((224,224)),
		torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(p=0.2),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.RandomAffine(45),
		torchvision.transforms.RandomRotation(20),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	if config.model.dataset == "mnist":
		mnist = datasets.MNIST(root=config.data.mnist_path, download=True, transform=transform, train=is_train)
		# train-validation split
		train_mnist, valid_mnist = train_valid_split(mnist)
		train_loader = torch.utils.data.DataLoader(dataset=train_mnist, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
		valid_loader = torch.utils.data.DataLoader(dataset=valid_mnist, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
	if config.model.dataset == "svhn":
		svhn = datasets.SVHN(root=config.data.svhn_path, download=True, transform=transform, split=mode)
		train_svhn, valid_svhn = train_valid_split(svhn)
		train_loader = torch.utils.data.DataLoader(dataset=train_svhn, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
		valid_loader = torch.utils.data.DataLoader(dataset=valid_svhn, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
	if config.model.dataset == "cifar10":
		svhn = datasets.CIFAR10(root=config.data.cifar10_path, download=True, transform=transforms, train=is_train)
		train_cifar, valid_cifar = train_valid_split(svhn)
		train_loader = torch.utils.data.DataLoader(dataset=train_cifar, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
		valid_loader = torch.utils.data.DataLoader(dataset=valid_cifar, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=config.data.num_workers)
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


def train_valid_split(ds, split_fold=10, random_seed=None):
	'''
	This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
	efficiently.
	:return:
	'''
	if random_seed != None:
		np.random.seed(random_seed)
	
	dslen = len(ds)
	indices = list(range(dslen))
	valid_size = dslen // split_fold
	np.random.shuffle(indices)
	train_mapping = indices[valid_size:]
	valid_mapping = indices[:valid_size]
	train = GenHelper(ds, dslen - valid_size, train_mapping)
	valid = GenHelper(ds, valid_size, valid_mapping)
	
	return train, valid

