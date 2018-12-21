import _pickle as pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D


class PlotMeasurement():
	def __init__(self):
		self.measure_list = []
		self.measure_name = None
	
	def plot_train_test_loss(self, color="b", marker="P"):
		plt.Figure()
		plt.title('{} {} score'.format(self.fname, self.measure_name), fontsize=18)
		x_range = np.linspace(1, len(self.measure_list) - 1, len(self.measure_list))
		
		measure = plt.plot(x_range, self.measure_list, color=color, marker=marker, label=self.measure_name, linewidth=2)
		plt.legend(handler_map={measure: HandlerLine2D(numpoints=1)})
		plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
		plt.yscale('linear')
		plt.xlabel('Epoch')
		plt.ylabel('Score')
		plt.grid()
		plt.show()
		
		name_figure = "classifier_results_seed_{}/classifier_MMinfoGAN_{}_{}".format(self.seed, self.fname, self.measure_name)
		pickle.dump(self.measure_list, open("{}.pkl".format(name_figure), 'wb'))
		plt.savefig(name_figure + ".png")
		plt.close()
	
	@staticmethod
	def MMgeneral_plot_from_pkl(groupby="", PATH=None,title=None):
		import glob
		param_list = dict()
		files_list = defaultdict(list)
		dirs = [d for d in glob.iglob(PATH)]
		
		for dir in dirs:
			for f in glob.iglob("{}/classifier*{}*.pkl".format(dir, groupby)):
				fname = f.split("/")[-1]
				name_split = fname.split("_")
				mu = name_split[5]
				sigma = name_split[7]
				ndist = name_split[9]
				param_list[fname] = ("$\Sigma={},\mu={}$".format(sigma, mu))
				print(fname, f)
				try:
					np_max = np.max(pickle.load(open(f, "rb")))
					# np_max = pickle.load(open(f, "rb"))[-1]
					files_list[fname].append(np_max)
				except Exception as e:
					print("ERROR:{}\n{}".format(f, e))
		
		means = []
		std_errs = []
		for key in files_list.keys():
			current_experiment = files_list[key]
			num_experiments = len(current_experiment)
			if num_experiments > 4:
				print(key)
				print(np.mean(current_experiment, axis=0))
				print(np.std(current_experiment, axis=0) / num_experiments)
				means.append(np.mean(current_experiment, axis=0))
				std_errs.append(np.std(current_experiment, axis=0) / num_experiments)
			elif key in param_list.keys():
				del param_list[key]
		
		fig, ax = plt.subplots()
		models = set(param_list.values())
		title = title
		print("means", means)
		print(models)
		ax.set_title(title, fontsize=10)
		x_pos = np.arange(len(models))
		ax.bar(x_pos, means, yerr=std_errs, align='center', alpha=0.5, ecolor='black', capsize=10)
		ax.set_ylabel('Accuracy')
		ax.set_xticks(x_pos)
		ax.set_xticklabels(models)
		plt.xticks(rotation=90)
		ax.set_ylim([0.5, 0.63])
		# ax.set_title('Prior')
		ax.yaxis.grid(True)
		
		# Save the figure and show
		plt.tight_layout()
		
		plt.ylabel("Accuracy Score")
		plt.grid(True)
		plt.show()
		plt.savefig(title + ".png")
		plt.close()
	
	@staticmethod
	def MMgeneral_plot_from_pkl_comparison(groupby="",PATH=None):
		import glob
		param_list = defaultdict()
		files_list = defaultdict(list)
		dirs = [d for d in glob.iglob(PATH)]
		
		l = "fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_ndist_3,fashion-mnist_MultivariateGaussianSampler_mu_0.8_sigma_0.2_ndist_5,fashion-mnist_MultivariateGaussianSampler_mu_1.0_sigma_0.25_ndist_10,fashion-mnist_GaussianSample_mu_0.0_sigma_0.2_ndist_10,fashion-mnist_UniformSample_mu_0.0_sigma_0.15_ndist_10"
		tmp = l.split(",")
		for t in tmp:
			for dir in dirs:
				for f in glob.iglob("{}/classifier*{}*.pkl".format(dir, t)):
					fname = f.split("/")[-1]
					tmp = fname.split("_")
					sampler = tmp[3]
					mu = tmp[5]
					sigma = tmp[7]
					ndist = tmp[9]
					if sampler == "MultivariateGaussianSampler":
						param_list[fname] = ("{} modalities".format(ndist))
						try:
							np_max = np.max(pickle.load(open(f, "rb")))
							# np_max = pickle.load(open(f, "rb"))[-1]
							files_list[fname].append(np_max)
						except Exception as e:
							print("ERROR:{}\n{}".format(f, e))
					elif sampler == "GaussianSample":
						param_list[fname] = ("1d Gaussian".format(sigma, mu))
						try:
							np_max = np.max(pickle.load(open(f, "rb")))
							# np_max = pickle.load(open(f, "rb"))[-1]
							files_list[fname].append(np_max)
						except Exception as e:
							print("ERROR:{}\n{}".format(f, e))
					elif sampler == "UniformSample":
						param_list[fname] = "Uniform"
						try:
							np_max = np.max(pickle.load(open(f, "rb")))
							# np_max = pickle.load(open(f, "rb"))[-1]
							files_list[fname].append(np_max)
						except Exception as e:
							print("ERROR:{}\n{}".format(f, e))
		
		means = []
		std_errs = []
		keylist = files_list.keys()
		keylist = sorted(keylist)
		for key in keylist:
			current_experiment = files_list[key]
			num_experiments = len(current_experiment)
			if num_experiments > 4:
				means.append(np.mean(current_experiment, axis=0))
				std_errs.append(np.std(current_experiment, axis=0) / num_experiments)
			elif key in param_list.keys():
				del param_list[key]
		fig, ax = plt.subplots()
		
		models = set(param_list.values())
		title = 'MMinfoGAN comparison'
		# ax.set_title(title, fontsize=10)
		x_pos = np.arange(len(models))
		ax.bar(x_pos, means, yerr=std_errs, align='center', alpha=0.5, ecolor='black', capsize=10)
		ax.set_ylabel('Accuracy')
		ax.set_xticks(x_pos)
		ax.set_xticklabels(['Uniform', '1d Gaussian', '3 modalities', '5 modalities', '10 modalities'])
		# plt.xticks(rotation=90)
		ax.set_ylim([0.5, 0.63])
		# ax.set_title('Prior')
		ax.yaxis.grid(True)
		
		# Save the figure and show
		plt.tight_layout()
		
		plt.ylabel("Accuracy Score")
		plt.grid(True)
		plt.show()
		plt.savefig(title + ".png")
		plt.close()
