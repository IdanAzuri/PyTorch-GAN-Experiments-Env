import os
import sys

import numpy as np
import torch
from hbconfig import Config
from torch.autograd import Variable
from torchvision.utils import save_image

from one_shot_aug.module import Discriminator, Generator
from . import utils


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class OneShotAug():
	model_name = "OneShotAug"
	os.makedirs('images', exist_ok=True)
	D_PATH = f"{Config.train.model_dir}/discriminator"
	
	def __init__(self):
		self.tensorboard = utils.TensorBoard(Config.train.model_dir)
		self.discriminator = Discriminator()
		self.generator = Generator()
	
	def train_fn(self, criterion, d_optimizer, g_optimizer=None, resume=True):
		self.auxiliary_criterion = criterion
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
		train_loader, valid_loader = data_loader
		d_prev_loss = np.infty
		for curr_step_count, (images, labels) in enumerate(train_loader):
			step_count = self.prev_step_count + curr_step_count + 1  # init value
			self.batch_size = images.shape[0]
			
			# # Adversarial ground truths
			# valid_labels = Variable(FloatTensor(self.batch_size, 1).fill_(1.0), requires_grad=False)
			# fake_labels = Variable(FloatTensor(self.batch_size, 1).fill_(0.0), requires_grad=False)
			
			# Converting to Variable type
			labels = Variable(labels.type(LongTensor))
			images = Variable(images)
			# save previous state - if needed turn back
			self.save_model_params(self.D_PATH, self.discriminator)
			# Train the discriminator
			d_loss = self._train_discriminator(self.discriminator, images, labels)
			# load weights only if worse
			if d_loss < d_prev_loss:
				d_prev_loss = d_loss
				self.discriminator.load_state_dict(torch.load(self.D_PATH))
			
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
	
	def _train_discriminator(self, discriminator, real_images, real_labels):
		discriminator.zero_grad()
		auxiliary_loss = self.auxiliary_criterion
		# adversarial_loss = self.adversarial_criterion
		
		# Sample again from the generator
		# z = Variable(FloatTensor(np.random.normal(0, 1, (self.batch_size, Config.model.z_dim))))
		augmented_images = self.generator(real_images)
		
		# real_pred, real_aux = discriminator(real_images)
		# d_real_loss = (adversarial_loss(real_pred, valid_labels) + auxiliary_loss(real_aux, real_labels)) / 2
		
		# Loss for fake images
		predicted_labels = discriminator(augmented_images)
		d_loss = auxiliary_loss(predicted_labels, real_labels)
		
		d_loss.backward()
		self.d_optimizer.step()
		
		# # Calculate discriminator accuracy
		# pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
		# gt = np.concatenate([real_labels.data.cpu().numpy(), fake_labels.data.cpu().numpy()], axis=0)
		# d_acc = np.mean(np.argmax(pred, axis=1) == gt)
		
		return d_loss
	
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
	
	def sample_image(self, n_row, batches_done):
		"""Saves a grid of generated digits ranging from 0 to n_classes"""
		# Sample noise
		z = Variable(FloatTensor(Config.random.normal(0, 1, (n_row ** 2, Config.latent_dim))))
		# Get labels ranging from 0 to n_classes for n rows
		labels = np.array([num for _ in range(n_row) for num in range(n_row)])
		labels = Variable(LongTensor(labels))
		gen_imgs = self.generator(z, labels)
		save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)
	
	def build_criterion(self):
		auxiliary_loss = torch.nn.CrossEntropyLoss()
		return auxiliary_loss
	
	def save_model_params(self, path, model):
		torch.save(model.state_dict(), path)
	
	def build_optimizers(self, discriminator, generator):
		d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=Config.train.d_learning_rate, betas=Config.train.optim_betas)
		g_optimizer = torch.optim.Adam(discriminator.parameters(), lr=Config.train.g_learning_rate, betas=Config.train.optim_betas)  # TODO remove this line
		
		return d_optimizer, g_optimizer
	
	def _add_summary(self, step, summary):
		for tag, value in summary.items():
			self.tensorboard.scalar_summary(tag, value, step)
