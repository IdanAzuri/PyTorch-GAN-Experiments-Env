#!/usr/bin/env python

import argparse
import atexit

import utils
from data_loader import *
from miniimagenet_loader import read_dataset
from model import Model
DATA_DIR = 'data/miniimagenet'

def main(mode):
	model = Model(mode)
	model_func = model.model_builder()
	
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
	args = parser.parse_args()
	
	# Print Config setting
	Config(args.config)
	print("Config: ", Config)
	if Config.get("description", None):
		print("Config Description")
		for key, value in Config.description.items():
			print(f" - {key}: {value}")

	torch.manual_seed(Config.model.seed)
	torch.cuda.manual_seed_all(Config.model.seed)
	np.random.seed(Config.model.seed)
	# After terminated Notification to Slack
	atexit.register(utils.send_message_to_slack, config_name=args.config)
	
	main(args.mode)
