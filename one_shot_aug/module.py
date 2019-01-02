import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from hbconfig import Config


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
