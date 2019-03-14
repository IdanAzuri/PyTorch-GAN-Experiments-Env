from hbconfig import Config

from acgan import ACGAN
from gan import GAN
from one_shot_aug import OneShotAug


class Model():
	TRAIN_MODE = "train"
	EVALUATE_MODE = "evaluate"
	PREDICT_MODE = "predict"
	
	def __init__(self, mode):
		self.mode = mode
	
	def model_builder(self):
		# load model
		global model
		models = [ACGAN, GAN, OneShotAug]
		for current_model in models:
			if current_model.model_name == Config.model.name:
				model = current_model()
				break
		criterion = model.build_criterion()
		if self.mode == self.TRAIN_MODE:
			optimizers = model.build_optimizers(model.meta_net)
			
			return model.train_fn(criterion, optimizers)
		elif self.mode == self.EVALUATE_MODE:
			return model.evaluate_model()
		elif self.mode == self.PREDICT_MODE:
			return model.predict(criterion)
		else:
			raise ValueError(f"unknown mode: {self.mode}")
	
	def build_metric(self):
		pass
