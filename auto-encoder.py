from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import argparse

# import matplotlib.pyplot as plt
import torch.nn as nn

from data_loader import *
from miniimagenet_loader import AutoEncoder
from one_shot_aug import show_images_no_labels


batch_size = 64  # Number of samples in each batch
epoch_num = 5  # Number of epochs to train the network
lr = 0.001  # Learning rate

# class AutoEncoder(nn.Module):
# 	def __init__(self):
# 		super(AutoEncoder, self).__init__()
# 		self.path_to_save = f"ae/model"
# 		self.use_cuda= False
# 		# conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
# 		# self.n_filters = 32
# 		# self.layer1 = _conv_layer(3, self.n_filters, 3, 1, 0)
# 		# self.layer2 = _conv_layer(self.n_filters, self.n_filters // 2, 3, 1, 0)
# 		# self.layer3 = _conv_layer(self.n_filters // 2, self.n_filters // 4, 3, 1, 0)
# 		# # self.emb = nn.Sequential(nn.Linear(512, 512))
# 		# self.emb = nn.Sequential(nn.Linear(5408, 5408))
#
# 		# # deconv layers: (in_channel size, out_channel size, kernel_size, stride, padding, output_padding)
# 		# self.deconv1 = _conv_transpose_layer(self.n_filters // 4, self.n_filters // 2, 3, stride=3, padding=1, output_padding=1)
# 		# self.deconv2 = _conv_transpose_layer(self.n_filters // 2, self.n_filters, 3, stride=2, padding=2, output_padding=1)
# 		# self.deconv3 = _conv_transpose_layer(self.n_filters, 3, 3, stride=2, padding=3, output_padding=1)
#
# 		self.encoder = nn.Sequential(
# 			nn.Conv2d(1, 32, kernel_size=3, padding=1),
# 			nn.ReLU(True),
# 			nn.MaxPool2d(2),
#
# 			nn.Conv2d(32, 32, kernel_size=3, padding=1),
# 			nn.ReLU(True),
# 			nn.MaxPool2d(2),
#
# 			nn.Conv2d(32, 32, kernel_size=3, padding=1),
# 			nn.ReLU(True),
# 			nn.MaxPool2d(2),
#
# 			nn.Conv2d(32, 32, kernel_size=3, padding=1),
# 			nn.ReLU(True),
# 			nn.MaxPool2d(2),
#
# 			nn.Conv2d(32, 32, kernel_size=3, padding=1),
# 			nn.ReLU(True),
# 			nn.MaxPool2d(2)
# 			)
#
# 		self.decoder = nn.Sequential(
#
# 			nn.functional.interpolate(mode='bilinear', scale_factor=2),
# 			nn.ConvTranspose2d(32, 32, kernel_size=3),
# 			nn.ReLU(True),
#
# 			Interpolate(mode='bilinear', scale_factor=2),
# 			nn.ConvTranspose2d(32, 32, kernel_size=3),
# 			nn.ReLU(True),
#
# 			Interpolate(mode='bilinear', scale_factor=2),
# 			nn.ConvTranspose2d(32, 32, kernel_size=3),
# 			nn.ReLU(True),
#
# 			Interpolate(mode='bilinear', scale_factor=2),
# 			nn.ConvTranspose2d(32, 32, kernel_size=3),
# 			nn.ReLU(True),
#
# 			Interpolate(mode='bilinear', scale_factor=2),
# 			nn.ConvTranspose2d(32, 1, kernel_size=3),
# 			nn.ReLU(True),
#
# 			nn.Sigmoid()
# 			)
#
# 		mkdir_p(self.path_to_save)
# 		if self.use_cuda:
# 			self = self.cuda()
# 		summary(self, (3, 224, 224))
# 	def forward(self, x):
# 		print("Start Encode: ", x.shape)
# 		x = self.encoder(x)
# 		print("Finished Encode: ", x.shape)
# 		x = self.decoder(x)
# 		print("Finished Decode: ", x.shape)
# 		return x
# 	# def forward(self, x):
# 		# the autoencoder has 3 con layers and 3 deconv layers (transposed conv). All layers but the last have ReLu
# 		# activation function
# 		# x = self.layer1(x)
# 		# x = self.layer2(x)
# 		# x = self.layer3(x)
# 		# x = x.view(x.size()[0], -1)
# 		# x = self.emb(x)
# 		# x = x.reshape(-1, 8, 26, 26)
# 		# x = self.deconv1(x)
# 		# x = self.deconv2(x)
# 		# self.decoded = self.deconv3(x)
# 		# x = torch.tanh(self.decoded)
# 		# return x.reshape((-1, 3, 84, 84))
#
# 	def load_saved_model(self, path, model):
# 		latest_path = find_latest(path + "/")
# 		if latest_path is None:
# 			return 0, model
#
# 		checkpoint = torch.load(latest_path)
#
# 		step_count = checkpoint['step_count']
# 		state_dict = checkpoint['net']
# 		# if dataparallel
# 		# if "module" in list(state_dict.keys())[0]:
# 		try:
# 			new_state_dict = OrderedDict()
# 			for k, v in state_dict.items():
# 				name = k[7:]  # remove 'module.' of dataparallel
# 				new_state_dict[name] = v
#
# 			model.load_state_dict(new_state_dict)
# 			model.optimizer.load_state_dict(checkpoint['optimizer'])
# 		except:
# 			# else:
# 			model.load_state_dict(checkpoint['net'])
# 			if model.optimizer is not None:
# 				model.optimizer.load_state_dict(checkpoint['optimizer'])
#
# 		print(f"Load checkpoints...! {latest_path}")
# 		return step_count, model
#
# 	def set_optimizer(self):
# 		self.optimizer = Adam(self.parameters(), lr=lr)
#
# 	def save_checkpoint(self, step, max_to_keep=3):
# 		sorted_path = get_sorted_path(self.path_to_save)
# 		for i in range(len(sorted_path) - max_to_keep):
# 			os.remove(sorted_path[i])
#
# 		full_path = os.path.join(self.path_to_save, f"ae_{step}.pkl")
# 		torch.save({"step_count": step, 'net': self.state_dict(), 'optimizer': self.optimizer.state_dict(), }, full_path)
# 		print(f"Save checkpoints...! {full_path}")


# class Interpolate(nn.Module):
# 	def __init__(self, mode, scale_factor):
# 		super(Interpolate, self).__init__()
# 		self.interp = nn.functional.interpolate
# 		self.size = scale_factor
# 		self.mode = mode
#
# 	def forward(self, x):
# 		x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
# 		return x

# class AutoEncoderConv(nn.Module):
# 	def __init__(self):
# 		super(AutoEncoderConv, self).__init__()
#
# 		self.encoder = nn.Sequential(
# 			nn.Conv2d(1, 32, kernel_size=3, padding=1),
# 			nn.ReLU(True),
# 			nn.MaxPool2d(2),
#
# 			nn.Conv2d(32, 32, kernel_size=3, padding=1),
# 			nn.ReLU(True),
# 			nn.MaxPool2d(2),
#
# 			nn.Conv2d(32, 32, kernel_size=3, padding=1),
# 			nn.ReLU(True),
# 			nn.MaxPool2d(2),
#
# 			nn.Conv2d(32, 32, kernel_size=3, padding=1),
# 			nn.ReLU(True),
# 			nn.MaxPool2d(2),
#
# 			nn.Conv2d(32, 32, kernel_size=3, padding=1),
# 			nn.ReLU(True),
# 			nn.MaxPool2d(2)
# 			)
#
# 		self.decoder = nn.Sequential(
#
# 			nn.functional.interpolate(mode='bilinear', scale_factor=2),
# 			nn.ConvTranspose2d(32, 32, kernel_size=3),
# 			nn.ReLU(True),
#
# 			Interpolate(mode='bilinear', scale_factor=2),
# 			nn.ConvTranspose2d(32, 32, kernel_size=3),
# 			nn.ReLU(True),
#
# 			Interpolate(mode='bilinear', scale_factor=2),
# 			nn.ConvTranspose2d(32, 32, kernel_size=3),
# 			nn.ReLU(True),
#
# 			Interpolate(mode='bilinear', scale_factor=2),
# 			nn.ConvTranspose2d(32, 32, kernel_size=3),
# 			nn.ReLU(True),
#
# 			Interpolate(mode='bilinear', scale_factor=2),
# 			nn.ConvTranspose2d(32, 1, kernel_size=3),
# 			nn.ReLU(True),
#
# 			nn.Sigmoid()
# 			)
#
# 	def forward(self, x):
# 		print()
# 		print("Start Encode: ", x.shape)
# 		x = self.encoder(x)
# 		print("Finished Encode: ", x.shape)
# 		x = self.decoder(x)
# 		print("Finished Decode: ", x.shape)
# 		return x
# def show():
# 	global ep, batch_img, batch_label
# 	# ------ test the trained network ----- #
# 	# read a batch of test data containing 50 samples
# 	for ep in range(1):  # epochs loop
# 		for batch_img, batch_label in valid_dataset:  # batches loop
# 			# input = resize_batch(batch_img)
#
# 			# pass the test samples to the network and get the reconstructed samples (output of autoencoder)
# 			recon_img = ae(batch_img)
# 	# transfer the outputs and the inputs from GPU to CPU and convert into numpy array
# 	recon_img = recon_img.data.cpu().numpy()
# 	batch_img = batch_img.data.cpu().numpy()
# 	# roll the second axis so the samples follow the matplotlib order (batch, height, width, channels)
# 	# (batch, channels, height, width) --> (batch, height, width, channels)
# 	recon_img = np.moveaxis(recon_img, 1, -1)
# 	batch_img = np.moveaxis(batch_img, 1, -1)
# 	# plot the reconstructed images and their ground truths (inputs)
# 	plt.title('Reconstructed Images')
# 	rows = 1
# 	columns = 5
# 	for i in range(1, 5):
# 		plt.subplot(5, 1, i + 1)
# 		plt.imshow(recon_img[i, ..., 0])
# 	plt.figure(2)
# 	plt.title('Input Images')
# 	for i in range(5):
# 		plt.subplot(5, 1, i + 1)
# 		plt.imshow(batch_img[i, ..., 0])
# 	plt.show()


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
	ae.use_cude = is_cuda
	if is_cuda:
		ae.cuda()
	# define the loss (criterion) and create an optimizer
	criterion = nn.MSELoss()
	resume = True
	if resume:
		epoch, ae = ae.load_saved_model(ae.path_to_save, ae)
		print(f"Model has been loaded epoch:{epoch}, path:{ae.path_to_save}")
	else:
		epoch = 0
	for epoch in range(0, 30):  # epochs loop
		for batch_idx, (batch_img, batch_label) in enumerate(train_dataset):  # batches loop
			if is_cuda:
				batch_img = batch_img.cuda()
				batch_label = batch_label.cuda()
			output = ae(batch_img)
			# show_images_no_labels(output,batch_idx,"awe")
			# show_images_no_labels(batch_img,batch_idx,"original")
			loss = criterion(output, batch_img)  # calculate the loss
			if batch_idx % 50 == 0:
				print(f'batch_idx:{batch_idx} loss: ', loss.data.item())
			ae.optimizer.zero_grad()
			loss.backward()  # calculate the gradients (backpropagation)
			ae.optimizer.step()  # update the weights
			if batch_idx % 1000 == 0:
				ae.save_checkpoint(f"ep{epoch}_idx{batch_idx}")  # show()
				print(f"Saved in {ae.path_to_save}")
