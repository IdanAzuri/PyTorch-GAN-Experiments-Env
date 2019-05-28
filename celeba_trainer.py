import os

import pandas as pd
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms , models
from tqdm import tqdm


lr = 0.0002
max_epoch = 10000
batch_size = 7
image_size = 64
log_step = 100
IMAGE_PATH = '/cs/labs/daphna/daphna/data/celeba/celeba64_2000_combine'
SAMPLE_PATH = '../'
identity_celeb_a_csv = '/cs/labs/daphna/daphna/data/celeba/identity_CelebA.csv'


class FaceLandmarksDataset(Dataset):
	"""Face Landmarks dataset."""
	
	def __init__(self , csv_file , root_dir , transform=None , is_train=True):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		identities = pd.read_csv(csv_file , " " , header=None , engine='python')
		identities.columns = ["fname" , "id"]
		identities.sort_values(by=["id"] , inplace=True)
		if is_train:
			identities_train = identities.loc[identities['id'] <= 2400]
			identities_test = identities.loc[(identities['id'] > 2400) & (identities['id'] < 4000)]
			identities_test = identities_test.groupby('id').first()
			self.ids = pd.concat([identities_train , identities_test] , sort=False)
			self.ids = self.ids.reset_index()
		else:
			identities_test = identities.loc[(identities['id'] > 4000) & (identities['id'] < 5000)]
			self.ids = identities_test.reset_index()
		self.root_dir = root_dir
		self.transform = transform
		self.num_classes = identities_train.id.nunique()
	
	def __len__(self):
		return len(self.ids)
	
	def __getitem__(self , idx):
		img_name = os.path.join(self.root_dir , self.ids.loc[idx , "fname"])
		image = Image.open(img_name).convert('RGB')
		ids = self.ids.loc[idx , "id"].astype('int')
		if self.transform:
			image = self.transform(image)
		
		return (image , ids)


if not os.path.exists(SAMPLE_PATH):
	os.makedirs(SAMPLE_PATH)
transform = transforms.Compose([transforms.Resize((image_size , image_size)) ,  # transforms.Resize(image_size),
                                transforms.ToTensor() , transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))])
train_dataset = FaceLandmarksDataset(identity_celeb_a_csv , IMAGE_PATH , transform)
test_dataset = FaceLandmarksDataset(identity_celeb_a_csv , IMAGE_PATH , transform , is_train=False)
train_data_loader = DataLoader(dataset=train_dataset , batch_size=batch_size , shuffle=True , num_workers=8 , drop_last=True)
test_data_loader = DataLoader(dataset=test_dataset , batch_size=batch_size , shuffle=True , num_workers=8 , drop_last=True)


def conv(ch_in , ch_out , k_size , stride=2 , pad=1 , bn=True):
	layers = []
	layers.append(nn.Conv2d(ch_in , ch_out , k_size , stride , pad))
	if bn:
		layers.append(nn.BatchNorm2d(ch_out))
	nn.MaxPool2d()
	return nn.Sequential(*layers)


class PretrainedClassifier(nn.Module):
	def __init__(self , num_classes):
		# create model
		super().__init__()
		arch = 'resnet18'
		model_dir = "pre_resnet18_celebA"
		n_classes = num_classes
		dataset = "celebA_64"
		self.arch = arch
		self.path_to_save = f"{model_dir}/model"
		print("=> creating model '{}'".format(arch))
		print("=> using pre-trained model '{}'".format(arch))
		
		model = models.__dict__[arch](pretrained=True)
		epoch , classifier = self.load_saved_model(model_dir , model)
		model = classifier
		print(f"Model has been loaded epoch:{epoch}, path:{self.path_to_save}")
		for param in model.parameters():
			param.requires_grad = False
		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs , n_classes)
		self.model = model
		
		self.title = dataset + '-' + arch
	
	def forward(self , inputs):
		outputs = self.model(inputs)
		
		return outputs


# denormalization : [-1,1] -> [0,1]
# normalization : [0,1] -> [-1,1]
def denorm(x):
	out = (x + 1) / 2
	return out.clamp(0 , 1)


def train_model(model , dataloaders , dataset_sizes , criterion , optimizer , scheduler , use_gpu , num_epochs=1000):
	since = time.time()
	
	best_model_wts = model.state_dict()
	best_acc = 0.0
	
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch , num_epochs - 1))
		print('-' * 10)
		
		# Each epoch has a training and validation phase
		for phase in ['train' , 'val']:
			if phase == 'train':
				scheduler.step()
				model.train(True)  # Set model to training mode
			else:
				model.train(False)  # Set model to evaluate mode
			
			running_loss = 0.0
			running_corrects = 0
			
			# Iterate over data.
			for data in tqdm(dataloaders[phase]):
				# get the inputs
				inputs , labels = data
				
				# wrap them in Variable
				if use_gpu:
					inputs = Variable(inputs.cuda())
					labels = Variable(labels.cuda())
				else:
					inputs , labels = Variable(inputs) , Variable(labels)
				
				# zero the parameter gradients
				optimizer.zero_grad()
				
				# forward
				outputs = model(inputs)
				if type(outputs) == tuple:
					outputs , _ = outputs
				_ , preds = torch.max(outputs.data , 1)
				loss = criterion(outputs , labels)
				
				# backward + optimize only if in training phase
				if phase == 'train':
					loss.backward()
					optimizer.step()
				
				# statistics
				running_loss += loss.data[0]
				running_corrects += torch.sum(preds == labels.data)
			
			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects / dataset_sizes[phase]
			
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase , epoch_loss , epoch_acc))
		
		print()
	
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))
	print('Best test Acc: {:4f}'.format(best_acc))
	
	# load best model weights
	model.load_state_dict(best_model_wts)
	return model


"""
Some of the links which I found useful.
https://discuss.pytorch.org/t/freeze-the-learnable-parameters-of-resnet-and-attach-it-to-a-new-network/949/9
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import argparse


parser = argparse.ArgumentParser(description='Training a pytorch model to classify different plants')
parser.add_argument('-idl' , '--input_data_loc' , help='' , default='data/training_data')
parser.add_argument('-mo' , '--model_name' , default="resnet18")
parser.add_argument('-p' , '--pre' , default=True , action='store_false' , help='Bool type')
parser.add_argument('-ep' , '--epochs' , default=100 , type=int)
parser.add_argument('-b' , '--batch_size' , default=64 , type=int)
parser.add_argument('-is' , '--input_shape' , default=224 , type=int)
parser.add_argument('-sl' , '--save_loc' , default="models/")
parser.add_argument("-g" , '--use_gpu' , default=True , action='store_false' , help='Bool type gpu')
parser.add_argument("-p" , '--use_parallel' , default=True , action='store_false' , help='Bool type to use_parallel')
parser.add_argument("-mx" , '--mixup' , default=True , action='store_true' , help='Use mixup data augementation')

args = parser.parse_args()

dataloaders = [train_data_loader , test_data_loader]
dataset_sizes = [train_dataset.__len__() , test_dataset.__len__()]

print("[Load the model...]")
model_conv = PretrainedClassifier(num_classes=2400)
if args.use_parallel:
	print("[Using all the available GPUs]")
	model_conv = nn.DataParallel(model_conv , device_ids=[0 , 1])

print("[Using CrossEntropyLoss...]")
criterion = nn.CrossEntropyLoss()

print("[Using small learning rate with momentum...]")
optimizer_conv = optim.SGD(list(filter(lambda p: p.requires_grad , model_conv.parameters())) , lr=0.001 , momentum=0.9)

print("[Creating Learning rate scheduler...]")
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv , step_size=7 , gamma=0.1)

print("[Training the model begun ....]")
print(args.mixup , args.mixup_alpha)
model_ft = train_model(model_conv , dataloaders , dataset_sizes , criterion , optimizer_conv , exp_lr_scheduler , args.use_gpu , num_epochs=args.epochs)

print("[Save the best model]")
model_save_loc = args.save_loc + args.model_name + ".pth"
