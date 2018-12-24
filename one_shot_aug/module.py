import torch.nn as nn
from hbconfig import Config


class Generator():
	# def __init__(self):
	
	def __call__(self, img):
		return img


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		
		def discriminator_block(in_filters, out_filters, bn=True):
			"""Returns layers of each discriminator block"""
			block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
			if bn:
				block.append(nn.BatchNorm2d(out_filters, 0.8))
			return block
		
		self.conv_blocks = nn.Sequential(*discriminator_block(Config.model.channels, Config.model.conv1, bn=False), *discriminator_block(Config.model.conv1, Config.model.conv2),
		                                 *discriminator_block(Config.model.conv2, Config.model.conv3), *discriminator_block(Config.model.conv3, Config.model.conv4))
		
		# The height and width of downsampled image
		ds_size = int(Config.data.image_size / 2) ** 4
		
		# Output layers
		self.output_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, Config.model.n_classes), nn.Softmax())
	
	def forward(self, image):
		out = self.conv_blocks(image)
		out = out.view(out.shape[0], -1)
		label = self.output_layer(out)
		
		return label
