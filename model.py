from hbconfig import Config

from acgan import ACGAN
from gan import GAN


class Model():
	TRAIN_MODE = "train"
	EVALUATE_MODE = "evaluate"
	PREDICT_MODE = "predict"
	
	def __init__(self, mode):
		self.mode = mode
	
	def model_builder(self):
		#load model
		global model
		models = [ACGAN, GAN]
		for current_model in models:
			if current_model.model_name == Config.model.name:
				model=current_model()
				break
		if self.mode == self.TRAIN_MODE:
			criterion = model.build_criterion()
			d_optimizer, g_optimizer = model.build_optimizers(model.discriminator, model.generator)
			
			return model.train_fn(criterion, d_optimizer, g_optimizer)
		elif self.mode == self.EVALUATE_MODE:
			return model.evaluate_model()
		elif self.mode == self.PREDICT_MODE:
			return model.predict()
		else:
			raise ValueError(f"unknown mode: {self.mode}")
		
	
	def build_metric(self):
		pass
