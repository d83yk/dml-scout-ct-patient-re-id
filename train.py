

from collections import OrderedDict
import shutil
import socket

from torchvision import transforms as T

import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

import timm
from timm.scheduler import CosineLRScheduler#from torch.optim import lr_scheduler

import socket
import datetime

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import subprocess
import pathlib

from lib.metrics import AdaCos #forked from https://github.com/4uiiurz1/pytorch-adacos

from lib.sam import SAM
from lib.utility.log import Log
from lib.utility.bypass_bn import enable_running_stats, disable_running_stats
from lib.RandomTransversalScaling import RandomTransversalScaling

class YMScout_Dataset(Dataset):

	def __init__(self,	image_root, csv_file, transform=None):
		self.image_root = image_root
		self.tags = pd.read_csv(csv_file)
		self.transform = transform

		self.classes = list(OrderedDict.fromkeys(self.tags.label))
		self.domain_classes = list(OrderedDict.fromkeys(self.tags.device))
		self.targets = self.tags.label
		self.domains = self.tags.device
		self.numbers = self.tags.number
		self.num_followup = self.tags.followup
		self.paths = self.tags.path

	def __len__(self):
		return len(self.tags)

	def __getitem__(self, idx):
		image = Image.open(os.path.join(self.image_root, self.tags.path[idx])).convert('L') # from PIL import Image
		if self.transform:
			image = self.transform(image)

		target  = self.targets[idx]
		domain = self.domains[idx]
		return image, target, domain

class Transform(object):
	def __init__(self, pixelsize, augment=False):
		self.augment = augment
		self.pixelsize=pixelsize

	def augmentation(self):
		augment = []
		augment.append(T.RandomPerspective(distortion_scale=0.1, p=0.1, interpolation=T.InterpolationMode.BICUBIC, fill=0))
		augment.append(T.RandomRotation(degrees=5, interpolation=T.InterpolationMode.BICUBIC))
		augment.append(RandomTransversalScaling(plus_width=10, p=0.25, interpolation=T.InterpolationMode.BICUBIC))
		return augment

	def augtrans(self):
		transform = []
		transform.append(T.Grayscale(num_output_channels=1))
		if self.augment:
			for a in self.augmentation():
				transform.append(a)
				
		transform.append(T.CenterCrop(self.pixelsize))
		transform.append(T.ToTensor())
		return T.Compose(transform)
				
	def __call__(self, img):
		transform = self.augtrans()
		return transform(img)

class Featurier(nn.Module):#resnetrs50
	def __init__(self, model_name, px_size):
		super(Featurier, self).__init__()
		self.px_size = px_size
		self.in_channels = 1
		self.featurier = timm.create_model(model_name=model_name, pretrained=False, in_chans=self.in_channels, num_classes=0)
		self.out_features = self.featurier.num_features

	def forward(self, x):
		x = x.expand(x.data.shape[0], self.in_channels, self.px_size[0], self.px_size[1])
		x = self.featurier(x)
		x = x.view(x.size(0), -1)
		return x
	
class Classifier(nn.Module):
	def __init__(self, in_features, hide_classes, n_classes):
		super(Classifier, self).__init__()
		self.in_channels = 1
		self.in_features = in_features
		self.classifier = nn.Sequential(
			nn.Linear(in_features, hide_classes),
			torch.nn.BatchNorm1d(hide_classes),
			nn.ReLU(),
			nn.Linear(hide_classes, in_features)
			)
		self.metric = AdaCos(num_features=self.in_features, num_classes=n_classes)

	def forward(self, x, t=None):
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		x = self.metric(x, t)
		return x

def training(args, featurier, classifier, criterion, train_loader, optimizer, log):
	featurier.train()
	classifier.train()
	log.train(len_dataset=len(train_loader))

	for i, batch in enumerate(train_loader):#train_progress:
		images, target_labels, domain_labels = (b.to(args.device, non_blocking=True) for b in batch)# get the inputs

		# first forward-backward step
		enable_running_stats(featurier)

		outputs	= classifier(featurier(images), target_labels).squeeze()
		loss = criterion(outputs, target_labels)
		loss.mean().backward()
		optimizer.first_step(zero_grad=True)

		# second forward-backward step
		disable_running_stats(featurier)
		criterion(classifier(featurier(images), target_labels).squeeze(), target_labels)\
			.mean().backward()
		optimizer.second_step(zero_grad=True)

		with torch.no_grad():
			correct = torch.argmax(outputs.data, 1) == target_labels
			log(featurier, loss.cpu(), correct.cpu(), optimizer.param_groups[0]["lr"])
			#scheduler.step(epoch)				

	return 	log.epoch_state["loss"]/log.epoch_state["steps"],\
			log.epoch_state["accuracy"]/log.epoch_state["steps"]


def validating(args, featurier, classifier, criterion, valid_loader, log):
	featurier.eval()
	classifier.eval()

	log.eval(len_dataset=len(valid_loader))

	with torch.no_grad():

		for i, batch in enumerate(valid_loader):
			images, target_labels, domain_labels = (b.to(args.device, non_blocking=True) for b in batch)# get the inputs

			outputs	= classifier(featurier(images), target_labels).squeeze()
			loss = criterion(outputs, target_labels)#.sum().item()
			correct = torch.argmax(outputs.data, 1) == target_labels
			log(featurier, loss.cpu(), correct.cpu())

	return 	log.epoch_state["loss"]/log.epoch_state["steps"],\
			log.epoch_state["accuracy"]/log.epoch_state["steps"]


#save current model
def saveCurrentModel(cpt, epoch, featurier, classifier, optimizer, scheduler, train_loss, valid_loss, best_loss):
	#save Current model (checkpoint)
	torch.save({'epoch': epoch,
				'featurier': featurier.state_dict(),
				'classifier': classifier.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
				'train_loss': train_loss,
				'valid_loss': valid_loss,
				'best_loss': best_loss,
				},
				cpt)

	
def main():
	# Training settings
	parser = argparse.ArgumentParser(description='Metric learning Example')
	#paths
	parser.add_argument('--seed', type=int, default=1, metavar='S',	help='random seed (default: 1)')
	parser.add_argument('--main_root', default=os.path.dirname(__file__),	help='Dir for model, .csv')#os.getcwd())
	parser.add_argument('--model_root', default='',	help='root for model')#	args.model_root	= os.path.join(args.main_root, args.model_path)
	parser.add_argument('--model_path', default='model',	help='Dir for model')
	parser.add_argument('--current_model', default='currentmodel.cpt',	help='For Saving the current Model')
	parser.add_argument('--best_model', default='bestmodel.cpt',	help='For Saving the current Model')
	parser.add_argument('--log_csv', default='log.csv',	help='For Saving Log')
	parser.add_argument('--image_root', default=r'C:\Studies\DML\Scout\Dataset\PNG\CT3')
	parser.add_argument('--trainset_csv', default=r'dataset\trainset.csv',	help='For Save train dataset list')
	parser.add_argument('--validset_csv', default=r'dataset\validset.csv',	help='For Save valid dataset list')
	#hyperparameter
	#AdaCos
	parser.add_argument("--train_patients", default=732, type=int)
	#optimizer - SAM
	parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
	parser.add_argument("--momentum", default=0.8, type=float, help="SGD Momentum.")
	parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
	parser.add_argument("--weight_decay", default=2.e-3, type=float, help="L2 weight decay.")
	#loss function - smooth cross entropy
	parser.add_argument("--label_smoothing", default=0.05, type=float, help="Use 0.0 for no label smoothing.") #smooth cross entropy
	#scheduler cosine scheduler
	parser.add_argument('--t_in_epochs', default=True, type=bool)
	parser.add_argument('--cycle_decay', default=0.9, type=float)
	parser.add_argument('--cycle_limit', default=1, type=int)
	parser.add_argument('--warmup_t', default=0, type=float)
	parser.add_argument('--warmup_lr_init', default=1e-2, type=float)
	parser.add_argument('--lr', default=5.e-3, type=float)
	parser.add_argument('--lr_min', default=5.e-4, type=float)
	parser.add_argument('--epochs', type=int, default=300, metavar='N',	help='number of epochs to train')
	# featurier
	parser.add_argument('--arch', default='tf_efficientnetv2_l', help='model architecture')
	parser.add_argument('--pixelsize', default=[384,224], type=int)
	#classifier
	parser.add_argument('--hide_classes', default=2048, type=int, help='dimention of hidden features')
	#mini batch
	parser.add_argument('--train_batch_size', type=int, default=20, help='L:32 M:56 @ 24GB mem.')
	parser.add_argument('--valid_batch_size', type=int, default=128,help='input batch size for testing (default: 1000)')
	parser.add_argument('--num_workers', default=os.cpu_count(), type=int)
	parser.add_argument('--pin_memory', default=True, type=bool)
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

	args = parser.parse_args()

	args.model_path	= os.path.join(args.model_path, args.arch[-1], str(args.train_patients), str(args.pixelsize[0])+"x"+str(args.pixelsize[1]), os.path.split(__file__)[-1].split('.')[0])
	args.model_root	= os.path.join(args.main_root, args.model_path)
	os.makedirs(args.model_root, exist_ok=True)
	shutil.copyfile(__file__, os.path.join(args.model_root, os.path.basename(__file__)))
	args.current_model	= os.path.join(args.model_root, args.current_model)
	args.best_model		= os.path.join(args.model_root, args.best_model)
	args.log_csv		= os.path.join(args.model_root, args.log_csv)
	args.trainset_csv	= os.path.join(args.main_root, args.trainset_csv)
	args.validset_csv	= os.path.join(args.main_root, args.validset_csv)

	torch.manual_seed(args.seed)

	train_kwargs = {'batch_size': args.train_batch_size, 'drop_last': False}
	valid_kwargs = {'batch_size': args.valid_batch_size, 'drop_last': False}
	cuda_kwargs  = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory, 'shuffle': True}
	train_kwargs.update(cuda_kwargs)
	valid_kwargs.update(cuda_kwargs)

	train_tr = Transform(augment=True,	pixelsize=args.pixelsize)
	valid_tr = Transform(augment=False,	pixelsize=args.pixelsize)

#	# import train dataset
	print(f'train dataset ---')
	train_set = YMScout_Dataset(args.image_root, args.trainset_csv, train_tr)
	#number of images
	print(f'num_images: {len(train_set.numbers)}')
	#number of patients
	num_patients = len(train_set.classes)
	print(f'num_patients: {num_patients}')
	num_domains = len(train_set.domain_classes)
	print(f'num_domains: {num_domains}')
	train_loader = DataLoader(train_set, **train_kwargs)

#	# import valid dataset
	print(f'valid dataset ---')
	valid_set = YMScout_Dataset(args.image_root, args.validset_csv, valid_tr)
	#number of images
	print(f'num_images: {len(valid_set.numbers)}')
	#number of patients
	num_patients = len(valid_set.classes)
	print(f'num_patients: {num_patients}')
	num_domains = len(valid_set.domain_classes)
	print(f'num_domains: {num_domains}')
	valid_loader = DataLoader(valid_set, **valid_kwargs)

	featurier	= Featurier(model_name=args.arch, px_size=args.pixelsize).to(args.device)
	classifier 	= Classifier(in_features=featurier.out_features, hide_classes=args.hide_classes, n_classes=args.train_patients).to(args.device)

	criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smoothing)
	optimizer = SAM(
			list(featurier.parameters()) + list(classifier.parameters()),
			torch.optim.SGD, rho=args.rho, adaptive=args.adaptive, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

	scheduler = CosineLRScheduler(	optimizer,
									t_initial= args.epochs / args.cycle_limit, 
									lr_min=args.lr_min, 
									cycle_limit= args.cycle_limit, 
									cycle_decay=args.cycle_decay, 
									t_in_epochs=args.t_in_epochs
									)#cycle cosine decay

	col_label = ['epoch', 'lr', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'best_loss', 'elapsed_time']
	elapsed_log = pd.DataFrame(columns=col_label)  
	if not (os.path.isfile(args.log_csv)):
		elapsed_log.to_csv(args.log_csv, mode='w', index=False, header=True)
	else:
		elapsed_log = pd.read_csv(args.log_csv)

	initial_epoch = 1
	best_loss = float('inf')
	if (os.path.isfile(args.current_model)):
		checkpoint = torch.load(args.current_model, torch.device(args.device))
		featurier.load_state_dict(checkpoint['featurier'])
		classifier.load_state_dict(checkpoint['classifier'])
		scheduler.load_state_dict(checkpoint['scheduler'])

		initial_epoch = checkpoint['epoch']+1 if 'epoch' in checkpoint else 1
		best_loss = checkpoint['best_loss'] if 'best_loss' in checkpoint else float('inf')	

		optimizer.load_state_dict(checkpoint['optimizer'])
		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(args.device, non_blocking=True)

	log = Log(log_each=10, initial_epoch=initial_epoch)
	log.best_loss = best_loss

	print(f'load scores ---')
	print(f'initial epoch: {initial_epoch}')
	print('best validation loss: %.4f' %(best_loss))

	cudnn.benchmark = True

	for epoch in range(initial_epoch, args.epochs +1):
		best = False
		train_loss, train_acc1 = training(args, featurier, classifier, criterion, train_loader, optimizer, log)
		scheduler.step(epoch)
		valid_loss, valid_acc1 = validating(args, featurier, classifier, criterion, valid_loader, log)
		elapsed_time = datetime.datetime.now()
		
		lr = optimizer.param_groups[0]["lr"]
		#Save Best model
		if best_loss > valid_loss and epoch > 1:
			best_loss = valid_loss
			best = True

		log_each = pd.DataFrame(data=[[epoch, lr, train_loss, train_acc1, valid_loss, valid_acc1, best_loss, elapsed_time]], columns=col_label)
		log_each.to_csv(args.log_csv, mode='a', index=False, header=False)
		elapsed_log = pd.concat([elapsed_log, log_each], ignore_index=True, axis=0)

#		scheduler.step(epoch)
		saveCurrentModel(args.current_model, epoch, featurier, classifier, optimizer, scheduler,  train_loss, valid_loss, best_loss)
		if best:
			shutil.copyfile(args.current_model, args.best_model)

	subprocess.run(['python', os.path.join(args.main_root, 'test.py'),\
					'--main_root='+ args.main_root,\
					'--model_path='+ args.model_path,\
					'--model_pth='+ pathlib.Path(args.best_model).name,\
					'--train_patients='+ str(args.train_patients),\
					'--device='+ args.device
					])

if __name__ == '__main__':
	main()
