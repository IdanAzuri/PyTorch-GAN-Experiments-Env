import os
import sys
from copy import deepcopy

import numpy as np
import torch
from hbconfig import Config
from one_shot_aug.module import Discriminator, Generator
from torch.autograd import Variable
from torchvision.utils import save_image

from . import utils


# FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
outerstepsize0 = 0.1  # stepsize of outer optimization, i.e., meta-optimization


class OneShotAug():
	model_name = "OneShotAug"
	os.makedirs('images', exist_ok=True)
	D_PATH = f"{Config.train.model_dir}/discriminator"
	
	def __init__(self):
		use_cuda = True if torch.cuda.is_available() else False
		self.device = torch.device("cuda" if use_cuda else "cpu")
		self.tensorboard = utils.TensorBoard(Config.train.model_dir)
		self.discriminator = Discriminator().to(self.device)
		self.generator = Generator()
	
	def train_fn(self, criterion, d_optimizer, g_optimizer=None, resume=True):
		self.loss_criterion = criterion
		self.d_optimizer = d_optimizer
		# self.g_optimizer = g_optimizer
		
		if resume:
			self.prev_step_count, self.discriminator, self.d_optimizer = utils.load_saved_model(self.D_PATH, self.discriminator, self.d_optimizer)
		return self._train
	
	def _train(self, data_loader):
		while True:
			d_loss, g_loss = self._train_epoch(data_loader)
	
	def _train_epoch(self, data_loader):
		global d_loss, g_loss, step_count
		train_loader, validation_loader = data_loader
		print(len(train_loader))
		
		for curr_step_count, (images, labels) in enumerate(train_loader):
			images, labels = images.to(self.device), labels.to(self.device)
			step_count = self.prev_step_count + curr_step_count + 1  # init value
			self.batch_size = images.shape[0]
			
			# Converting to Variable type
			# labels = Variable(labels)
			# images = Variable(images)
			weights_before = deepcopy(self.discriminator.state_dict())
			# Train the discriminator
			weights_factor_by_validation = 0.
			for i in range(3):
				d_loss, validation_loss, validation_accurcy = self._train_discriminator(self.discriminator, images, labels, validation_loader)
				weights_factor_by_validation += validation_accurcy
			# TODO consider multiple weights_factor_by_validation by the weights after - maybe bucketing that
			print("previous_validation loss ={}, current_validation_loss={}".format(weights_factor_by_validation, d_loss))
			weights_after = self.discriminator.state_dict()
			outerstepsize = outerstepsize0 * (1 - curr_step_count /len(train_loader.dataset))  # linear schedule
			self.discriminator.load_state_dict({name: weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize for name in weights_before})
			# Step Verbose & Tensorboard Summary
			if step_count % Config.train.verbose_step_count == 0:
				g_loss_data = g_loss.data[0].item()
				d_loss_data = d_loss.data[0].item()
				loss = d_loss_data + g_loss_data
				self._add_summary(step_count, {"Loss": loss, "D_Loss": d_loss_data, "G_Loss": g_loss_data})
				
				print(f"Step {step_count} - Loss: {loss} (D: {d_loss_data}, G: {g_loss_data})")
			
			# Save model parameters
			if step_count % Config.train.save_checkpoints_steps == 0:
				utils.save_checkpoint(step_count, self.D_PATH, self.discriminator,
				                      self.d_optimizer)  # utils.save_checkpoint(step_count, self.G_PATH, self.generator, self.g_optimizer)
			
			if step_count >= Config.train.train_steps:
				sys.exit()
		
		self.prev_step_count = step_count
		return d_loss, g_loss
	
	def _train_discriminator(self, discriminator, real_images, real_labels, validation_set):
		discriminator.zero_grad()
		loss = self.loss_criterion
		# adversarial_loss = self.adversarial_criterion
		
		# Sample again from the generator
		# z = Variable(FloatTensor(np.random.normal(0, 1, (self.batch_size, Config.model.z_dim))))
		augmented_images = self.generator(real_images)
		
		# real_pred, real_aux = discriminator(real_images)
		# d_real_loss = (adversarial_loss(real_pred, valid_labels) + loss(real_aux, real_labels)) / 2
		
		# Loss for fake images
		predicted_labels = discriminator(augmented_images)
		d_loss = loss(predicted_labels, real_labels)
		
		d_loss.backward()
		self.d_optimizer.step()
		
		# validation statistics
		validation_loss = 0.0
		validation_accurcy = 0
		for valid_images, valid_labels in validation_set:
			valid_images, valid_labels = valid_images.to(self.device), valid_labels.to(self.device)
			valid_predicted_labels = discriminator(valid_images)
			validation_loss += loss(valid_predicted_labels, valid_labels).item()
			pred = valid_predicted_labels.max(1, keepdim=True)[1] # get the index of the max log-probability
			validation_accurcy += pred.eq(valid_labels.view_as(pred)).sum().item()
		
		return d_loss, validation_loss, validation_accurcy
	
	# def _train_generator(self, generator, gen_labels):
	# 		pass
	# generator.zero_grad()
	# auxiliary_loss = self.auxiliary_criterion
	# adversarial_loss = self.adversarial_criterion
	#
	# # Samplefrom the generator and get output from discriminator
	# z = Variable(FloatTensor(np.random.normal(0, 1, (self.batch_size, Config.model.z_dim))))
	# fake_images = self.generator(z, gen_labels)
	# validity, pred_gen_label = self.discriminator(fake_images)
	#
	# # Loss measures generator's ability to fool the discriminator
	# g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_gen_label, gen_labels))
	#
	# g_loss.backward()
	# self.g_optimizer.step()
	# return g_loss
	
	def evaluate_model(self):
		pass
	
	def predict(self):
		pass
	

	
	def build_criterion(self):
		auxiliary_loss = torch.nn.CrossEntropyLoss()
		return auxiliary_loss
	
	def save_model_params(self, model):
		self.weights_before = deepcopy(model.state_dict())
		# torch.save(model.state_dict(), path)
		return self.weights_before
	
	def build_optimizers(self, discriminator, generator):
		d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=Config.train.d_learning_rate, betas=Config.train.optim_betas)
		g_optimizer = torch.optim.Adam(discriminator.parameters(), lr=Config.train.g_learning_rate, betas=Config.train.optim_betas)  # TODO remove this line
		
		return d_optimizer, g_optimizer
	
	def _add_summary(self, step, summary):
		for tag, value in summary.items():
			self.tensorboard.scalar_summary(tag, value, step)
