#!/usr/bin/env python

import argparse
import atexit
import random

import basic_utils
import utils
from data_loader import *
from miniimagenet_loader import read_dataset, AutoEncoder
from model import Model
DATA_DIR = 'data/miniimagenet'

def main(mode,seed):
	model = Model(mode)
	model_func = model.model_builder(seed)
	
	if mode == Model.TRAIN_MODE:
		# if Config.model.dataset == "miniimagenet":
		# 	train_and_valid_loaders = read_dataset(DATA_DIR) # train_set, val_set, test_set
		# else:
		train_and_valid_loaders = get_loader("train")
		model_func(train_and_valid_loaders)
	elif mode == Model.PREDICT_MODE:
		model_func
	else:
		raise ValueError(f"unknown mode: {Model.mode}")


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
	seed = args.seed
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	# After terminated Notification to Slack
	atexit.register(basic_utils.send_message_to_slack, config_name=args.config)
	
	
	train_dataset, valid_dataset = get_loader("train")
# define the autoencoder and move the network into GPU
	ae = AutoEncoder()
	ae.train()
	ae.set_optimizer()
	is_cuda = True if torch.cuda.is_available() else False
	if is_cuda:
		ae.cuda()
	# define the loss (criterion) and create an optimizer
	criterion = torch.nn.MSELoss()
	resume = False
	if resume:
		epoch, ae = ae.load_saved_model(ae.path_to_save, ae)
		print(f"Model has been loaded epoch:{epoch}, path:{ae.path_to_save}")
	else:
		epoch = 0
	for epoch in range(epoch, 100):  # epochs loop
		for batch_idx, (batch_img, batch_label) in enumerate(train_dataset):  # batches loop
			output = ae(batch_img)
			loss = criterion(output, batch_img)  # calculate the loss
			print(f'batch_idx:{batch_idx} loss: ', loss.data.item())
			ae.optimizer.zero_grad()
			loss.backward()  # calculate the gradients (backpropagation)
			ae.optimizer.step()  # update the weights
		ae.save_checkpoint(epoch)
	# main(args.mode,seed)
