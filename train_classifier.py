from __future__ import print_function

import matplotlib

from one_shot_aug import PretrainedClassifier


matplotlib.use('Agg')
import argparse

# import matplotlib.pyplot as plt

from data_loader import *


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
	
	classifier = PretrainedClassifier()
	c = classifier.train_model()
	
	# Save model
	c.save_state_dict('fine_tuned_best_model.pt')
