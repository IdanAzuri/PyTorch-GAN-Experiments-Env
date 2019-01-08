from math import floor

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from hbconfig import Config
from torch.nn import init


class Generator():
	# def __init__(self):
	
	def __call__(self, img):
		return img


class Discriminator(nn.Module):
	def __init__(self):
		# super(Discriminator, self).__init__()
		
		def discriminator_block(in_filters, out_filters, bn=True):
			"""Returns layers of each discriminator block"""
			block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
			if bn:
				block.append(nn.BatchNorm2d(out_filters, 0.8))
			return block
		
		super(Discriminator, self).__init__()
	
	# self.conv1 = nn.Conv2d(1, 20, 5, 1)
	# self.conv2 = nn.Conv2d(20, 50, 5, 1)
	# self.fc1 = nn.Linear(4 * 4 * 50, 500)
	# self.fc2 = nn.Linear(500, 10)
		
		
	def forward(self, image):
		# out = self.conv_blocks(image)
		# out = out.view(out.shape[0], -1)
		# label = self.output_layer(out)
		# return label
		x = F.relu(self.conv1(image))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4 * 4 * 50)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)


class PretrainedClassifier():
	def __init__(self):
		# create model
		arch = Config.model.arch
		self.arch = arch
		if Config.model.pretrained:
			print("=> using pre-trained model '{}'".format(arch))
			model = models.__dict__[arch](pretrained=True)
		elif arch.startswith('resnext'):
			model = models.__dict__[arch](baseWidth=Config.model.base_width, cardinality=Config.model.cardinality, )
		else:
			print("=> creating model '{}'".format(arch))
			model = models.__dict__[arch]()
		
		self.model = model
		self.title = Config.model.dataset + '-' + arch


def _conv_layer(n_input, n_output, k):
	"3x3 convolution with padding"
	seq = nn.Sequential(nn.Conv2d(n_input, n_output, kernel_size=k, stride=1, padding=1, bias=True),
	                    nn.BatchNorm2d(n_output),
	                    nn.LeakyReLU(True),
	                    nn.MaxPool2d(kernel_size=2, stride=2))
	if Config.model.use_dropout:  # Add dropout module
		list_seq = list(seq.modules())[1:]
		list_seq.append(nn.Dropout(Config.model.dropout))
		seq = nn.Sequential(*list_seq)
	return seq


class MiniImageNetModel(nn.Module):
	"""
	A model for Mini-ImageNet classification.
	"""
	
	def __init__(self):
		super(MiniImageNetModel, self).__init__()
		self.arch = "vgg_small"
		self.title = Config.model.dataset + '-' + self.arch
		self.n_filters = Config.model.filters
		# The height and width of downsampled image
		ds_size = Config.data.image_size // 2 ** 4
		
		self.layer1 = _conv_layer(Config.data.channels, self.n_filters, 3)
		self.layer2 = _conv_layer(self.n_filters, self.n_filters, 3)
		self.layer3 = _conv_layer(self.n_filters, self.n_filters, 3)
		self.layer4 = _conv_layer(self.n_filters, self.n_filters, 3)
		# self.out = nn.Sequential(nn.Linear(self.n_filters * ds_size ** 2, Config.model.n_classes), nn.Softmax())
		self.out = nn.Linear(self.n_filters * ds_size ** 2, Config.model.n_classes)
		
		# Initialize layers
		self.weights_init(self.layer1)
		self.weights_init(self.layer2)
		self.weights_init(self.layer3)
		self.weights_init(self.layer4)
	
	def weights_init(self, module):
		for m in module.modules():
			if isinstance(m, nn.Conv2d):
				init.xavier_uniform_(m.weight, gain=np.sqrt(2))
				init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
	
	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.view(x.size(0), -1)
		x = self.out(x)
		return x
