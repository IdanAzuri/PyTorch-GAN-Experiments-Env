from __future__ import print_function

import matplotlib


matplotlib.use('Agg')
import argparse

# import matplotlib.pyplot as plt
import torch.nn as nn

from data_loader import *
from miniimagenet_loader import AutoEncoder


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
	for epoch in range(0, 2):  # epochs loop
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
