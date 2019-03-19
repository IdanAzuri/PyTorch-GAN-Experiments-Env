import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from hbconfig import Config
from torch.autograd import Variable
from torch.nn import init


# class Discriminator(nn.Module):
# 	def __init__(self):
# 		# super(Discriminator, self).__init__()
#
# 		def discriminator_block(in_filters, out_filters, bn=True):
# 			"""Returns layers of each discriminator block"""
# 			block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
# 			if bn:
# 				block.append(nn.BatchNorm2d(out_filters, 0.8))
# 			return block
#
# 		super(Discriminator, self).__init__()
#
# 	# self.conv1 = nn.Conv2d(1, 20, 5, 1)
# 	# self.conv2 = nn.Conv2d(20, 50, 5, 1)
# 	# self.fc1 = nn.Linear(4 * 4 * 50, 500)
# 	# self.fc2 = nn.Linear(500, 10)
#
# 	def forward(self, image):
# 		# out = self.conv_blocks(image)
# 		# out = out.view(out.shape[0], -1)
# 		# label = self.output_layer(out)
# 		# return label
# 		x = F.relu(self.conv1(image))
# 		x = F.max_pool2d(x, 2, 2)
# 		x = F.relu(self.conv2(x))
# 		x = F.max_pool2d(x, 2, 2)
# 		x = x.view(-1, 4 * 4 * 50)
# 		x = F.relu(self.fc1(x))
# 		x = self.fc2(x)
# 		return F.log_softmax(x, dim=1)


class PretrainedClassifier(nn.Module):
	def __init__(self):
		# create model
		super().__init__()
		arch = Config.model.arch
		self.arch = arch
		print("=> creating model '{}'".format(arch))
		print("=> using pre-trained model '{}'".format(arch))
		# model = models.__dict__[arch](pretrained=True)
		if arch.startswith('densenet') or arch.startswith('inception'):
			model_conv = models.__dict__[arch](pretrained=Config.model.pretrained)
			num_ftrs = model_conv.classifier.in_features
			model_conv.classifier = nn.Linear(num_ftrs, Config.model.n_classes)
			self.model = model_conv
		elif arch.startswith('vgg'):
			model_conv = models.__dict__[arch](pretrained=Config.model.pretrained)
			# Number of filters in the bottleneck layer
			num_ftrs = model_conv.classifier[6].in_features
			# convert all the layers to list and remove the last one
			features = list(model_conv.classifier.children())[:-1]
			## Add the last layer based on the num of classes in our dataset
			features.extend([nn.Linear(num_ftrs, Config.model.n_classes)])
			## convert it into container and add it to our model class.
			model_conv.classifier = nn.Sequential(*features)
			model_conv.num_classes = Config.model.n_classes
			
			self.model = model_conv
		elif arch.startswith('resne'):
			model = models.__dict__[arch](pretrained=Config.model.pretrained)
			num_ftrs = model.fc.in_features
			model.fc = nn.Linear(num_ftrs, Config.model.n_classes)
			self.model = model
		
		self.title = Config.model.dataset + '-' + arch
	
	def forward(self, inputs):
		outputs = self.model(inputs)
		
		return outputs


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
		self.out = nn.Sequential(nn.Linear(self.n_filters * ds_size ** 2, Config.model.n_classes), nn.LogSoftmax(1))
		# self.out = nn.Linear(self.n_filters * ds_size ** 2, Config.model.n_classes)
		
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
		
	def point_grad_to(self, target_param, is_cuda, step_size):
		'''
		from reptile
		Set .grad attribute of each parameter to be proportional
		to the difference between self and target
		'''
		for p, target_p in zip(self.parameters(), target_param.parameters()):
			if p.grad is None:
				if is_cuda:
					p.grad = Variable(torch.zeros(p.size())).cuda()
				else:
					p.grad = Variable(torch.zeros(p.size()))
			p.grad.data.zero_()  # not sure this is required
			p.grad.data.add_(step_size * (p.data - target_p.data))
	
	def clone(self, use_cuda):
		clone = MiniImageNetModel()
		clone.load_state_dict(self.state_dict())
		if use_cuda:
			clone.cuda()
		return clone

def _conv_layer(n_input, n_output, k):
	"3x3 convolution with padding"
	seq = nn.Sequential(nn.Conv2d(n_input, n_output, kernel_size=k, stride=1, padding=1, bias=True), nn.BatchNorm2d(n_output), nn.LeakyReLU(True),
	                    nn.MaxPool2d(kernel_size=2, stride=2))
	if Config.model.use_dropout:  # Add dropout module
		list_seq = list(seq.modules())[1:]
		list_seq.append(nn.Dropout(Config.model.dropout))
		seq = nn.Sequential(*list_seq)
	return seq
