import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from torch.autograd import Variable
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms


lr = 0.0002
max_epoch = 10000
batch_size = 64
image_size = 64
log_step = 100
IMAGE_PATH = '/cs/labs/daphna/daphna/data/celeba/celeba64_2000'
SAMPLE_PATH = '../'
identity_celeb_a_csv = '/cs/labs/daphna/daphna/data/celeba/identity_CelebA.csv'


class FaceLandmarksDataset(Dataset):
	"""Face Landmarks dataset."""
	
	def __init__(self , csv_file , root_dir , transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir=root_dir
		identities = pd.read_csv(csv_file , " " , header=None,engine='python')
		identities.columns = ["fname" , "id"]
		identities.sort_values(by=["id"] , inplace=True)
		identities_train = identities.loc[identities['id'] <= 2400]
		identities_test = identities.loc[(identities['id'] > 2400) & (identities['id'] < 4401)]
		identities_test = identities_test.groupby('id').first()
		self.identities = pd.concat([identities_train , identities_test])
		# self.identities = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform
	
	def __len__(self):
		return len(self.identities)
	
	def __getitem__(self , idx):
		img_name = os.path.join(self.root_dir , self.identities.iloc[idx , "fname"])
		image = io.imread(img_name)
		identities = self.identities.iloc[idx , "id"].as_matrix()
		identities = identities.astype('int')
		sample = {'image': image , 'identities': identities}
		
		if self.transform:
			sample = self.transform(sample)
		
		return sample


if not os.path.exists(SAMPLE_PATH):
	os.makedirs(SAMPLE_PATH)
transform = transforms.Compose([transforms.Scale(image_size) ,  # transforms.Resize(image_size),
                                transforms.ToTensor() , transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))])
dataset = FaceLandmarksDataset(identity_celeb_a_csv , IMAGE_PATH , transform)
data_loader = DataLoader(dataset=dataset , batch_size=batch_size , shuffle=True , num_workers=8 , drop_last=True)


def conv(ch_in , ch_out , k_size , stride=2 , pad=1 , bn=True):
	layers = []
	layers.append(nn.Conv2d(ch_in , ch_out , k_size , stride , pad))
	if bn:
		layers.append(nn.BatchNorm2d(ch_out))
	return nn.Sequential(*layers)


class Discriminator(nn.Module):
	def __init__(self , image_size=128 , conv_dim=64):
		super(Discriminator , self).__init__()
		self.conv1 = conv(3 , conv_dim , 4 , bn=False)
		self.conv2 = conv(conv_dim , conv_dim * 2 , 4)
		self.conv3 = conv(conv_dim * 2 , conv_dim * 4 , 4)
		self.conv4 = conv(conv_dim * 4 , conv_dim * 8 , 4)
	
	# self.fc = conv(conv_dim * 8 , 1 , int(image_size / 16) , 1 , 0 , False)
	
	def forward(self , x):  # if image_size is 64, output shape is below
		out = F.leaky_relu(self.conv1(x) , 0.05)  # (?, 64, 32, 32)
		out = F.leaky_relu(self.conv2(out) , 0.05)  # (?, 128, 16, 16)
		out = F.leaky_relu(self.conv3(out) , 0.05)  # (?, 256, 8, 8)
		out = F.leaky_relu(self.conv4(out) , 0.05)  # (?, 512, 4, 4)
		# out = F.log_softmax(self.fc(out))
		return out.squeeze()


D = Discriminator(image_size)
D.cuda()
criterion = nn.CrossEntropyLoss().cuda()
d_optimizer = torch.optim.Adam(D.parameters() , lr=lr , betas=(0.5 , 0.999))


# denormalization : [-1,1] -> [0,1]
# normalization : [0,1] -> [-1,1]
def denorm(x):
	out = (x + 1) / 2
	return out.clamp(0 , 1)


try:
	D.load_state_dict(torch.load('discriminator.pkl'))
	print("\n-------------model restored-------------\n")
except:
	print("\n-------------model not restored-------------\n")
	pass

total_batch = len(data_loader.dataset) // batch_size
for epoch in range(max_epoch):
	for i , (images , labels) in enumerate(data_loader):
		# Build mini-batch dataset
		image = Variable(images).cuda()
		# Create the labels which are later used as input for the BCE loss
		lables = Variable(torch.ones(batch_size)).cuda()
		outputs = D(image)
		loss = criterion(outputs , lables)
		real_score = outputs
		
		# Backprob + Optimize
		D.zero_grad()
		loss.backward()
		d_optimizer.step()
		
		if (i + 1) % log_step == 0:
			print("Epoch [%d/%d], Step[%d/%d], d_loss: %.4f" % (epoch , max_epoch , i + 1 , total_batch , loss.item()))

torch.save(D.state_dict() , 'discriminator.pkl')
