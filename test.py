
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from pathlib import Path
import warnings

import shutil
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import timeit
import timm
from torchvision import transforms as T
import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class YMScout_Dataset(Dataset):

	def __init__(self,	image_root, csv_file):
		self.image_root = image_root
		self.tags = pd.read_csv(csv_file)
		self.tags.number	= LabelEncoder().fit_transform(self.tags.number)
		self.tags.label	= LabelEncoder().fit_transform(self.tags.label)
		self.tags.device	= LabelEncoder().fit_transform(self.tags.device)

		self.classes = list(OrderedDict.fromkeys(self.tags.label))
		self.domain_classes = list(OrderedDict.fromkeys(self.tags.device))
		self.targets = self.tags.label
		self.domains = self.tags.device
		self.vendor = self.tags.manufacture
		self.numbers = self.tags.number
		self.num_followup = self.tags.followup
		self.paths = self.tags.path

	def __len__(self):
		return len(self.tags)

	def __getitem__(self, idx):
		image = Image.open(os.path.join(self.image_root, self.tags.path[idx])).convert('L') # from PIL import Image
		target  = self.targets[idx]
		domain = self.domains[idx]
		vendor = self.vendor[idx]
		return image, target, domain, vendor

class Subset(Dataset):
	def __init__(self, dataset, indices, transform=None):
		self.dataset = dataset
		self.indices = indices
		self.transform = transform
		self.classes = list(OrderedDict.fromkeys(self.dataset.targets[self.indices]))

	def __getitem__(self, idx):
		img, target, domain, vendor = self.dataset[self.indices[idx]]
		if self.transform:
			img = self.transform(img)
		return img, target, domain, vendor

	def __len__(self):
		return len(self.indices)

class Transform(object):
	def __init__(self, pixelsize):
		self.pixelsize=pixelsize

	def augtrans(self):
		transform = []
		transform.append(T.Grayscale(num_output_channels=1))
		transform.append(T.CenterCrop(self.pixelsize))
		transform.append(T.ToTensor())
		return T.Compose(transform)
				
	def __call__(self, img):
		transform = self.augtrans()
		return transform(img)

class MyBatchSize(object):
	def __init__(self, max_batchsize):
		self.max_batchsize = max_batchsize

	def ge_common_divisor(self, arr1, arr2):#greatest_common divisor
		for n in sorted(arr1, reverse=True):
			if n in arr2:
				return n
		return 1

	def valid_divisors(self, max, arr):
		divisors = []
		for n in sorted(arr, reverse=False):
			if max < n:
				return divisors
			else:
				divisors.append(n)
		return divisors

	def make_divisors(self, n):
		lower_divisors , upper_divisors = [], []
		i = 1
		while i*i <= n:
			if n % i == 0:
				lower_divisors.append(i)
				if i != n // i:
					upper_divisors.append(n//i)
			i += 1
		return lower_divisors + upper_divisors[::-1]

	def __call__(self, num_baseline, num_followup):
		bl = self.valid_divisors(self.max_batchsize, self.make_divisors(num_baseline))
		fl = self.valid_divisors(self.max_batchsize, self.make_divisors(num_followup))
		return self.ge_common_divisor(bl, fl)

def eer(fpr,tpr,th):
	""" Returns equal error rate (EER) and the corresponding threshold. """
	fnr = 1-tpr
	abs_diffs = np.abs(fpr - fnr)
	min_idx = np.argmin(abs_diffs)
	eer = np.mean((fpr[min_idx], fnr[min_idx]))
	return eer, th[min_idx]

class Featurier(nn.Module):
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


def main():
	tic=timeit.default_timer()

	parser = argparse.ArgumentParser()
	parser.add_argument('--name', default='test',	help='model name: save folder')
	parser.add_argument('--prefix', default='',	help='model name: save folder prefix')
	parser.add_argument('--store', default='',	help='model name: save folder')
	
	parser.add_argument('--image_root', default=r'C:\Studies\DML\Scout\Dataset\PNG\CT3')
	parser.add_argument('--dataset_csv', default=r'dataset\testset.csv')
	parser.add_argument('--main_root', default=os.path.dirname(__file__))#os.getcwd())
	parser.add_argument('--model_path', default='based_table-wo')
	parser.add_argument('--model_pth', default='bestmodel.cpt')
	parser.add_argument('--pixelsize', default=[384,256], type=int)

	parser.add_argument('--arch', default='tf_efficientnetv2_l', help='*from args.model_path')
	parser.add_argument('--train_patients', default=732, type=int, help='patient num. on training *from args.model_path')
	parser.add_argument('--baseline_patients', default=1, type=int, help='patient num. verification')
	parser.add_argument('--followup_patients', default=1, type=int, help='patient num. verification')
	parser.add_argument('-b', '--batch_size', default=512,	type=int,	metavar='N', help='mini-batch size') 
	parser.add_argument('--num_workers', default=os.cpu_count(), type=int)
	parser.add_argument('--pin_memory', default=True, type=bool)
	parser.add_argument('--drop_last', default=False,	type=bool)
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	args = parser.parse_args()

	if args.device == 'cuda':
		torch.backends.cudnn.deterministic = True

	print("main_root: " + args.main_root)
	args.dataset_csv = os.path.join(args.main_root, args.dataset_csv)
	print("dataset_csv: " + args.dataset_csv)
	print("device: " + args.device)

	warnings.simplefilter('ignore')

#	# import dataset
#################################################################################
	all_dataset = YMScout_Dataset(image_root=args.image_root, csv_file=args.dataset_csv)

	bl_subset = all_dataset.numbers[all_dataset.num_followup==0].tolist()
	fu_subset = all_dataset.numbers[all_dataset.num_followup> 0].tolist()

	bl_dataset = Subset(all_dataset, bl_subset, Transform(pixelsize=args.pixelsize))
	fu_dataset = Subset(all_dataset, fu_subset, Transform(pixelsize=args.pixelsize))

	#number of patients
	args.baseline_patients = len(bl_dataset.classes), len(bl_dataset.indices)
	print(f'baseline_patients: {args.baseline_patients}')
	args.followup_patients = len(fu_dataset.classes), len(fu_dataset.indices)
	print(f'followup_patients: {args.followup_patients}')

	#calc batch_number
	print(str(args.batch_size), end=" -> ")
	MyBatch = MyBatchSize(args.batch_size)
	args.batch_size = MyBatch(args.baseline_patients[1],args.followup_patients[1])
	print(str(args.batch_size))

	img_table = pd.DataFrame(
					all_dataset.tags,
					columns=all_dataset.tags.columns[1:],
					)

	baseline_loader = DataLoader(
			bl_dataset, batch_size=args.batch_size,	shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
	followup_loader = DataLoader(
			fu_dataset, batch_size=args.batch_size,	shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)

	model = Featurier(model_name=args.arch, px_size=args.pixelsize).to(args.device)

	#identification
	device_list = ['ACC1:All','0=0','0=1','0=2','1=0','1=1','1=2','2=0','2=1','2=2']
	vendor_list = ['Diff','Same', 'S=S', 'S=G', 'G=S', 'G=G']
	score_matrix, label_matrix = [], []
	target1_matrix, target2_matrix = [], []
	dom1_matrix, dom2_matrix = [], []
	vendor1_matrix, vendor2_matrix = [], []

	with torch.no_grad():

		model.eval()	# switch to evaluate mode
		current_model = torch.load(os.path.join(args.main_root, args.model_path, args.model_pth))

		model.load_state_dict(current_model['featurier'])
		current_epoch = current_model['epoch']

		#save dataset and parameter
		args.prefix = str('{0:03d}'.format(current_epoch))
		args.store = os.path.join(args.main_root,args.model_path,os.path.basename(__file__), args.name)
		Path(args.store).mkdir(parents=True, exist_ok=True)
		baseline_csv = os.path.join(args.store, 'baseline-idx.csv')
		followup_csv = os.path.join(args.store, 'followup-idx.csv')
		args_txt	 = os.path.join(args.store, 'args.txt')
		
		#save args values
		with open(args_txt, 'w') as f:
			for arg in vars(args):
				print('%s: %s' %(arg, getattr(args, arg)), file=f)
		#save baseline and followup
		img_table.loc[fu_subset].to_csv(followup_csv)
		img_table.loc[bl_subset].to_csv(baseline_csv)

		f1v_list,f2v_list = list(),list()

		for (input2, target2, dom2, vendor2) in tqdm(followup_loader, total=len(followup_loader), desc=str(current_epoch)+"-"+args.model_pth):
			f2v = model(input2.to(args.device))
			f2v_list.append(f2v)
			target2_matrix.extend([ i2 for i2 in target2.tolist() ])
			dom2_matrix.extend([ i2 for i2 in dom2 ])
			vendor2_matrix.extend([ i2 for i2 in vendor2 ])
		f2v_list = torch.stack(f2v_list, dim=0).view(-1, model.out_features)

		for (input1, target1, dom1, vendor1) in tqdm(baseline_loader, total=len(baseline_loader), desc=str(current_epoch)+"-"+args.model_pth):
			f1v = model(input1.to(args.device))
			f1v_list.append(f1v)
			target1_matrix.extend([ i1 for i1 in target1.tolist() ])
			dom1_matrix.extend([ i1 for i1 in dom1 ])
			vendor1_matrix.extend([ i1 for i1 in vendor1 ])
		f1v_list = torch.stack(f1v_list, dim=0).view(-1, model.out_features)

		for j, f2v in tqdm(enumerate(f2v_list), total=len(target2_matrix), desc=str(current_epoch)+"-"+args.model_pth):
			f2v_repeat = f2v.repeat((len(target1_matrix), 1))
			score_matrix.insert(	j,
									F.cosine_similarity(
										f2v_repeat, 
										f1v_list,
										dim=1,
										eps=1e-5
										).cpu().detach().numpy().copy()
									)

	genuine_list,impostor_list = [],[]

	label_matrix  = np.zeros((len(target2_matrix),len(target1_matrix)), dtype=np.int16)
	rank_device = np.zeros((len(all_dataset.domain_classes)*len(all_dataset.domain_classes)+1, len(target2_matrix))) 
	cmc_device = np.empty((len(all_dataset.domain_classes)*len(all_dataset.domain_classes)+1, len(target1_matrix))) 
	device_str = ['']*len(target2_matrix)

	rank_vendor = np.zeros((len(vendor_list), len(target2_matrix)))
	cmc_vendor	= np.empty((len(vendor_list), len(target1_matrix)))
	same_vendor = [0]*len(target2_matrix)
	vendor_str = ['']*len(target2_matrix)


	for j, t2 in tqdm(enumerate(target2_matrix), total=len(target2_matrix), desc=str(current_epoch)+"-"+args.model_pth):
		for i, t1 in enumerate(target1_matrix):

			if score_matrix[j][i] != float('inf'):
				if t1 == t2:
					label_matrix[j,i] = 1
					genuine_list.append(score_matrix[j][i])

					rank_index = np.argsort(-score_matrix[j]).tolist() #ascending order temp[0] == top of cossim

					rank_device[0, j] = rank_index.index(i) + 1

					device_str[j] = str(dom1_matrix[i].item())+"="+str(dom2_matrix[j].item())
					device_idx = device_list.index(device_str[j])
					rank_device[device_idx, j] = rank_device[0,j]

					same_vendor[j] = 1 if (vendor1_matrix[i] == vendor2_matrix[j]) else 0
					rank_vendor[same_vendor[j], j] = rank_device[0,j]

					vendor_str[j] = str(vendor1_matrix[i])[0] +"="+ str(vendor2_matrix[j])[0]
					vendor_idx = vendor_list.index(vendor_str[j])
					rank_vendor[vendor_idx, j] = rank_device[0,j]

				else:
					label_matrix[j,i] = 0
					impostor_list.append(score_matrix[j][i])
			else:
				label_matrix[j,i] = -1

#verification - AUC and EER

	impostor_sample = np.array(random.sample(impostor_list, len(genuine_list)))
	genuine = np.asarray(genuine_list)
	genuine_impostor_score = [*genuine.tolist(), *impostor_sample.tolist()] # list
	genuine_impostor_label = [*[1]*len(genuine.tolist()),*[0]*len(genuine.tolist())] #list

	#AUC ROC
	roc_curve_png = os.path.join(args.store, 'roc_curve.png')
	fpr, tpr, th = roc_curve(genuine_impostor_label, genuine_impostor_score)
	auc_value = auc(fpr, tpr)
	eer_value, eer_pos = eer(fpr,tpr,th)
	plt.style.use('default')
	sns.set_palette('gray')
	plt.plot(fpr, tpr, marker='None')
	plt.xlabel('False rejection rate')
	plt.ylabel('Correct acceptance rate')
	plt.title(f'ROC Curve  AUC:{"{:.4f}".format(auc_value)}')
	plt.grid()
	plt.savefig(roc_curve_png)
	plt.clf()
	plt.close()

	#histogram
	histogram_png = os.path.join(args.store, 'histogram.png')
	sns.set()
	sns.set_style('whitegrid')
	sns.set_palette('Set1')
	ax = plt.figure().add_subplot(1, 1, 1)
	ax.hist(genuine,  bins=100, range=(0, 1), density=True, alpha=0.7)
	ax.hist(impostor_sample, bins=100, range=(0, 1), density=True, alpha=0.7)
	ax.set_xlabel('Cosine Similarity')
	ax.set_ylabel('Relative Frequency')
	plt.title('Genuine-Impostor histogram EER:%.4f %.4f' %(eer_value, eer_pos))
	#plt.show()
	plt.savefig('%s' %(histogram_png))
	plt.clf()
	plt.close()

	#pROC.tsv
	proc_tsv = os.path.join(args.store, 'pROC.tsv')
	pd.DataFrame({	'score':  genuine_impostor_score, 'genuine': genuine_impostor_label,}).to_csv(proc_tsv, sep='\t', index=False)

# ReID -Top-1 accuracy

	for j in range(len(target1_matrix)):
		for i, device in enumerate(rank_device):
			cmc_device[i,j] = np.count_nonzero(device==(j+1))
			cmc_device[i,j] += cmc_device[i,j-1] if j > 0 else 0
		for i, vendor in enumerate(rank_vendor):
			cmc_vendor[i,j] = np.count_nonzero(vendor==(j+1))
			cmc_vendor[i,j] += cmc_vendor[i,j-1] if j > 0 else 0

	#rank.csv
	rank_csv = os.path.join(args.store, 'rank.tsv')
	with open(rank_csv, 'w') as f_handle:
		np.savetxt(f_handle, [np.hstack(('series', np.array(target2_matrix)))], delimiter='\t', fmt="%s")
		np.savetxt(f_handle, [np.hstack(('score' , np.array(genuine)))], delimiter='\t', fmt="%s")
		np.savetxt(f_handle, [np.hstack(('device', np.array(device_str)))], delimiter='\t', fmt="%s")

		for i, device in enumerate(rank_device):
			np.savetxt(f_handle, [np.hstack((device_list[i], device))], delimiter='\t', fmt="%s")

		for i, vendor in enumerate(rank_vendor):
			np.savetxt(f_handle, [np.hstack((vendor_list[i], vendor))], delimiter='\t', fmt="%s")

		np.savetxt(f_handle, [np.hstack(('rank', [1,2,5,10,'Total']))], delimiter='\t', fmt="%s")

		rank_row = 4 + len(rank_device) + len(rank_vendor) #+3: vendor
		for r in range(4 + len(rank_vendor), rank_row):
			np.savetxt(f_handle, [np.hstack(('=A%d' %(r), [
													'=COUNTIFS(%d:%d,"<="&B$%d,%d:%d,">0.1")/COUNTIF(%d:%d,">0.1")' %(r,r,rank_row,r,r,r,r),
													'=COUNTIFS(%d:%d,"<="&C$%d,%d:%d,">0.1")/COUNTIF(%d:%d,">0.1")' %(r,r,rank_row,r,r,r,r),
													'=COUNTIFS(%d:%d,"<="&D$%d,%d:%d,">0.1")/COUNTIF(%d:%d,">0.1")' %(r,r,rank_row,r,r,r,r),
													'=COUNTIFS(%d:%d,"<="&E$%d,%d:%d,">0.1")/COUNTIF(%d:%d,">0.1")' %(r,r,rank_row,r,r,r,r),
													'=COUNTIF(%d:%d,">0.1")' %(r,r),
													]))], delimiter='\t', fmt="%s")

	#cmc.csv
	cmc_csv = os.path.join(args.store, 'cmc.tsv')
	with open(cmc_csv, 'w') as f_handle:
		np.savetxt(f_handle, [np.hstack(('series', range(1, args.baseline_patients[0]+1)))], delimiter='\t', fmt="%s")
		for i, device in enumerate(cmc_device):
			np.savetxt(f_handle, [np.hstack((device_list[i], device))], delimiter='\t', fmt="%s")
		for i, vendor in enumerate(cmc_vendor):
			np.savetxt(f_handle, [np.hstack((vendor_list[i], vendor))], delimiter='\t', fmt="%s")

	#show performance 
	print('AUC: %.4f  EER: %.4f' %(auc_value, eer_value))

	acc1_device = [cmc_device[0][0]/cmc_device[0][-1]] # all
	print('ACC1 All:%.4f' %(acc1_device[0]))

	for j in range(len(all_dataset.domain_classes)):
		for i in range(len(all_dataset.domain_classes)):
			d = i+j*len(all_dataset.domain_classes) + 1
			device = cmc_device[d]
			acc1_device.append(device[0]/device[-1])
			print('%s:%.4f' %(device_list[d], acc1_device[-1]), end=' ')
		print('')#to <br>


	acc1_vendor = []
	for d in range(len(vendor_list)):
		vendor = cmc_vendor[d]
		acc1_vendor.append(vendor[0]/vendor[-1])
		print('%s:%.4f' %(vendor_list[d], acc1_vendor[-1]), end=' ')

	#performance.tsv
	performance_tsv	= os.path.join(args.store, '%3d_acc1=%3f_auc=%3f_eer=%3f.tsv' %(current_epoch, acc1_device[0], auc_value, eer_value))
	with open(performance_tsv, 'w') as f_handle:
		np.savetxt(f_handle, [['pth', 'epoch', *device_list, *vendor_list, 'AUC', 'EER', 'EER_Pos','Genuine','G-SD', 'Impostor', 'I-SD']], delimiter='\t', fmt="%s")
		np.savetxt(f_handle,np.r_[[args.model_pth, current_epoch, *acc1_device, *acc1_vendor, auc_value, eer_value, eer_pos, np.average(genuine), np.std(genuine), np.average(impostor_sample), np.std(impostor_sample)]].reshape(1,-1), delimiter='\t', fmt="%s")

	toc=timeit.default_timer()
	print('elapsed time: %.1f sec.' %(toc - tic))

if __name__=='__main__':
	main()