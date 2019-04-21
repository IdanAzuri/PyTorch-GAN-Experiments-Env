from __future__ import print_function

from collections import OrderedDict

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
from data_loader import *
from hbconfig import Config
from torch.autograd import Variable
from torch.optim import Adam
from torchsummary import summary
from torchvision import transforms
from utils import _conv_layer, _conv_transpose_layer
from utils import get_sorted_path, find_latest, mkdir_p


batch_size = 64  # Number of samples in each batch
epoch_num = 5  # Number of epochs to train the network
lr = 0.001  # Learning rate


def resize_batch(imgs, img_size=84, img_dim=3):
	# A function to resize a batch of MNIST images to (32, 32)
	# Args:
	#   imgs: a numpy array of size [batch_size, 28 X 28].
	# Returns:
	#   a pytorch Variable of size [batch_size, 1, 32, 32].
	
	# reshape the sample to a batch of images in pytorch order (batch, channels, height, width)
	imgs = imgs.reshape((-1, img_dim, img_size, img_size))
	
	# resize the images to (32, 32)
	resized_imgs = np.zeros((imgs.shape[0], img_dim, img_size, img_size))
	for i in range(imgs.shape[0]):
		resized_imgs[i, 0, ...] = transforms.transform.resize(imgs[i, 0, ...], (img_size, img_size))
	
	resized_imgs = torch.from_numpy(resized_imgs).float()  # convert the numpy array into torch tensor
	if is_cuda:
		resized_imgs = Variable(resized_imgs).cuda()  # create a torch variable and transfer it into GPU
	return resized_imgs


class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()
		self.path_to_save = f"ae/model"
		# conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
		self.n_filters = 32
		self.layer1 = _conv_layer(3, self.n_filters, 3, 1, 0)
		self.layer2 = _conv_layer(self.n_filters, self.n_filters // 2, 3, 1, 0)
		self.layer3 = _conv_layer(self.n_filters // 2, self.n_filters // 4, 3, 1, 0)
		# self.emb = nn.Sequential(nn.Linear(512, 512))
		self.emb = nn.Sequential(nn.Linear(512, 512))
		
		# deconv layers: (in_channel size, out_channel size, kernel_size, stride, padding, output_padding)
		self.deconv1 = _conv_transpose_layer(self.n_filters // 4, self.n_filters // 2, 3, stride=3, padding=1, output_padding=1)
		self.deconv2 = _conv_transpose_layer(self.n_filters // 2, self.n_filters, 3, stride=2, padding=2, output_padding=1)
		self.deconv3 = _conv_transpose_layer(self.n_filters, 3, 3, stride=2, padding=3, output_padding=1)
		mkdir_p(self.path_to_save)
		summary(self.cuda(), (3, 84, 84))
	
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
		return x.reshape((-1, 3, 84, 84))
	
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
		self.optimizer = Adam(self.parameters(), lr=lr)
	
	def save_checkpoint(self, step, max_to_keep=3):
		sorted_path = get_sorted_path(self.path_to_save)
		for i in range(len(sorted_path) - max_to_keep):
			os.remove(sorted_path[i])
		
		full_path = os.path.join(self.path_to_save, f"ae_{step}.pkl")
		torch.save({"step_count": step, 'net': self.state_dict(), 'optimizer': self.optimizer.state_dict(), }, full_path)
		print(f"Save checkpoints...! {full_path}")


def show():
	global ep, batch_img, batch_label
	# ------ test the trained network ----- #
	# read a batch of test data containing 50 samples
	for ep in range(1):  # epochs loop
		for batch_img, batch_label in valid_dataset:  # batches loop
			# input = resize_batch(batch_img)
			
			# pass the test samples to the network and get the reconstructed samples (output of autoencoder)
			recon_img = ae(batch_img)
	# transfer the outputs and the inputs from GPU to CPU and convert into numpy array
	recon_img = recon_img.data.cpu().numpy()
	batch_img = batch_img.data.cpu().numpy()
	# roll the second axis so the samples follow the matplotlib order (batch, height, width, channels)
	# (batch, channels, height, width) --> (batch, height, width, channels)
	recon_img = np.moveaxis(recon_img, 1, -1)
	batch_img = np.moveaxis(batch_img, 1, -1)
	# plot the reconstructed images and their ground truths (inputs)
	plt.title('Reconstructed Images')
	rows = 1
	columns = 5
	for i in range(1, 5):
		plt.subplot(5, 1, i + 1)
		plt.imshow(recon_img[i, ..., 0])
	plt.figure(2)
	plt.title('Input Images')
	for i in range(5):
		plt.subplot(5, 1, i + 1)
		plt.imshow(batch_img[i, ..., 0])
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--config', type=str, default='config', help='config file name in config dir')
	parser.add_argument('--mode', type=str, default='train', help='Mode (train/test/train_and_evaluate)')
	parser.add_argument('--seed', type=int, default=0)
	args = parser.parse_args()
	
	# Print Config setting
	Config(args.config)
	print("Config: ", Config)
	if Config.get("description", None):
		print("Config Description")
		for key, value in Config.description.items():
			print(f" - {key}: {value}")
	train_dataset, valid_dataset = get_loader("train")
	# define the autoencoder and move the network into GPU
	ae = AutoEncoder()
	ae.train()
	ae.set_optimizer()
	is_cuda = True if torch.cuda.is_available() else False
	if is_cuda:
		ae.cuda()
	# define the loss (criterion) and create an optimizer
	criterion = nn.MSELoss()
	resume = False
	if resume:
		epoch, ae = ae.load_saved_model(ae.path_to_save, ae)
		print(f"Model has been loaded epoch:{epoch}, path:{ae.path_to_save}")
	else:
		epoch = 0
	for epoch in range(epoch, 30):  # epochs loop
		for batch_idx, (batch_img, batch_label) in enumerate(train_dataset):  # batches loop
			if is_cuda:
				batch_img=batch_img.cuda()
			output = ae(batch_img)
			loss = criterion(output, batch_img)  # calculate the loss
			if batch_idx % 50 == 0:
				print(f'batch_idx:{batch_idx} loss: ', loss.data.item())
			ae.optimizer.zero_grad()
			loss.backward()  # calculate the gradients (backpropagation)
			ae.optimizer.step()  # update the weights
			if batch_idx % 1000 == 0:
				ae.save_checkpoint(f"ep{epoch}_idx{batch_idx}")  # show()
				print(f"Saved in {ae.path_to_save}")