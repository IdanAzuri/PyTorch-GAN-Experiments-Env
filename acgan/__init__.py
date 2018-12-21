import os
import sys

import numpy as np
import torch
from hbconfig import Config
from torch.autograd import Variable
from torchvision.utils import save_image

from acgan.module import Discriminator, Generator
from gan import GAN
from . import utils


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class ACGAN(GAN):
	model_name = "ACGAN"
	os.makedirs('images', exist_ok=True)
	
	def __init__(self):
		super().__init__()
		self.discriminator = Discriminator()
		self.generator = Generator()
	
	def train_fn(self, criterion, d_optimizer, g_optimizer, resume=True):
		self.adversarial_criterion, self.auxiliary_criterion = criterion
		self.d_optimizer = d_optimizer
		self.g_optimizer = g_optimizer
		
		if resume:
			self.prev_step_count, self.discriminator, self.d_optimizer = utils.load_saved_model(self.D_PATH, self.discriminator, self.d_optimizer)
			_, self.generator, self.g_optimizer = utils.load_saved_model(self.G_PATH, self.generator, self.g_optimizer)
			if self.prev_step_count == 0:
				self.generator.apply(self.weights_init_normal)
		return self._train
	
	def _train_epoch(self, data_loader):
		train_loader, valid_loader = data_loader
		for curr_step_count, (images, real_labels) in enumerate(train_loader):
			step_count = self.prev_step_count + curr_step_count + 1  # init value
			self.batch_size = images.shape[0]
			
			# Adversarial ground truths
			valid_labels = Variable(FloatTensor(self.batch_size, 1).fill_(1.0), requires_grad=False)
			fake_labels = Variable(FloatTensor(self.batch_size, 1).fill_(0.0), requires_grad=False)
			
			# Converting to Variable type
			real_labels = Variable(real_labels.type(LongTensor))
			images = Variable(images)
			gen_labels = Variable(LongTensor(np.random.randint(0, Config.model.n_classes, self.batch_size)))
			
			# Train the generator
			g_loss = self._train_generator(self.generator, valid_labels, gen_labels)
			
			# Train the discriminator
			d_loss = self._train_acgan_discriminator(self.discriminator, images, valid_labels, gen_labels, fake_labels, real_labels)
			
			# Step Verbose & Tensorboard Summary
			if step_count % Config.train.verbose_step_count == 0:
				g_loss_data = g_loss.data[0].item()
				d_loss_data = d_loss.data[0].item()
				loss = d_loss_data + g_loss_data
				self._add_summary(step_count, {"Loss": loss, "D_Loss": d_loss_data, "G_Loss": g_loss_data})
				
				print(f"Step {step_count} - Loss: {loss} (D: { d_loss_data }, G: { g_loss_data })")
			
			# Save model parameters
			if step_count % Config.train.save_checkpoints_steps == 0:
				utils.save_checkpoint(step_count, self.D_PATH, self.discriminator, self.d_optimizer)
				utils.save_checkpoint(step_count, self.G_PATH, self.generator, self.g_optimizer)
			
			if step_count >= Config.train.train_steps:
				sys.exit()
		
		self.prev_step_count = step_count
		return d_loss, g_loss
	
	def _train_acgan_discriminator(self, discriminator, real_images, valid_labels, gen_labels, fake_labels, real_labels):
		discriminator.zero_grad()
		auxiliary_loss = self.auxiliary_criterion
		adversarial_loss = self.adversarial_criterion
		
		# Sample again from the generator
		z = Variable(FloatTensor(np.random.normal(0, 1, (self.batch_size, Config.model.z_dim))))
		fake_images = self.generator(z, gen_labels)
		
		real_pred, real_aux = discriminator(real_images)
		d_real_loss = (adversarial_loss(real_pred, valid_labels) + auxiliary_loss(real_aux, real_labels)) / 2
		
		# Loss for fake images
		fake_pred, fake_aux = discriminator(fake_images)
		d_fake_loss = (adversarial_loss(fake_pred, fake_labels) + auxiliary_loss(fake_aux, gen_labels)) / 2
		
		# Total discriminator loss
		d_loss = (d_real_loss + d_fake_loss) / 2
		
		d_loss.backward()
		self.d_optimizer.step()
		
		# # Calculate discriminator accuracy
		# pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
		# gt = np.concatenate([real_labels.data.cpu().numpy(), fake_labels.data.cpu().numpy()], axis=0)
		# d_acc = np.mean(np.argmax(pred, axis=1) == gt)
		
		return d_loss
	
	def _train_generator(self, generator, valid, gen_labels):
		generator.zero_grad()
		auxiliary_loss = self.auxiliary_criterion
		adversarial_loss = self.adversarial_criterion
		
		# Samplefrom the generator and get output from discriminator
		z = Variable(FloatTensor(np.random.normal(0, 1, (self.batch_size, Config.model.z_dim))))
		fake_images = self.generator(z, gen_labels)
		validity, pred_gen_label = self.discriminator(fake_images)
		
		# Loss measures generator's ability to fool the discriminator
		g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_gen_label, gen_labels))
		
		g_loss.backward()
		self.g_optimizer.step()
		
		return g_loss
	
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
		adversarial_loss = torch.nn.BCELoss()
		auxiliary_loss = torch.nn.CrossEntropyLoss()
		return adversarial_loss, auxiliary_loss
	
	@staticmethod
	def weights_init_normal(m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
		elif classname.find('BatchNorm2d') != -1:
			torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
			torch.nn.init.constant_(m.bias.data, 0.0)
