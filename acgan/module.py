import torch

import torch.nn as nn
from hbconfig import Config


class Generator(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.label_emb = nn.Embedding(Config.model.n_classes, Config.model.z_dim)
		
		self.init_size = Config.data.image_size // 4  # Initial size before upsampling
		self.l1 = nn.Sequential(nn.Linear(Config.model.z_dim, 128 * self.init_size ** 2))
		
		self.conv_blocks = nn.Sequential(nn.BatchNorm2d(128),
		                                 nn.Upsample(scale_factor=2),
		                                 nn.Conv2d(128, 128, 3, stride=1, padding=1),
		                                 nn.BatchNorm2d(128, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			                             nn.Upsample(scale_factor=2),
			                             nn.Conv2d(128, 64, 3, stride=1, padding=1),
			                             nn.BatchNorm2d(64, 0.8),
			                             nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, Config.model.channels, 3,
			          stride=1, padding=1), nn.Tanh())
	
	def forward(self, noise, labels):
		gen_input = torch.mul(self.label_emb(labels), noise)
		out = self.l1(gen_input)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		image = self.conv_blocks(out)
		return image
		


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		
		def discriminator_block(in_filters, out_filters, bn=True):
			"""Returns layers of each discriminator block"""
			block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
			         nn.LeakyReLU(0.2, inplace=True),
			         nn.Dropout2d(0.25)]
			if bn:
				block.append(nn.BatchNorm2d(out_filters, 0.8))
			return block
		
		self.conv_blocks = nn.Sequential(
			*discriminator_block(Config.model.channels, Config.model.conv1, bn=False),
			*discriminator_block(Config.model.conv1, Config.model.conv2),
			*discriminator_block(Config.model.conv2, Config.model.conv3),
			*discriminator_block(Config.model.conv3, Config.model.conv4))
		
		# The height and width of downsampled image
		ds_size = Config.data.image_size // 2 ** 4
		
		# Output layers
		self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
		self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, Config.model.n_classes), nn.Softmax())
	
		# The height and width of downsampled image
		ds_size = Config.data.image_size // 2 ** 4
		
		# Output layers
		self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
		self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, Config.model.n_classes), nn.Softmax())

	def forward(self, image):
		out = self.conv_blocks(image)
		out = out.view(out.shape[0], -1)
		validity = self.adv_layer(out)
		label = self.aux_layer(out)
		
		return validity, label
