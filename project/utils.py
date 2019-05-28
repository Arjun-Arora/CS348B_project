import numpy as np 
import torch
import csv
import skimage as sk
import os 
import sys
import glob
from sklearn.feature_extraction import image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
import torch.nn as nn
from skimage.measure import compare_ssim as ssim
sys.path.append("./models/")
from MWU_CNN import MW_Unet 


class dummyDataset(Dataset):
	'''
	This is a dummy dataset that just returns tensors of img_sz with ch channels
	Output format is: N,C,H,W 
	'''
	def __init__(self,ch=5,img_sz = (240,240),length = 10000):
		assert len(img_sz) == 2
		self.ch = ch
		self.img_sz = img_sz
		self.length = length
		self.sz = (self.ch,self.img_sz[0],self.img_sz[1])
	def __len__(self):
		return self.length
	def __getitem__(self,idx):
		input_img = np.random.randint(0,255,size=self.sz)
		target_img = np.random.randint(0,255,size=self.sz)
		sample = {'target':target_img,'input':input_img}
		return sample

if __name__ =="__main__":
	Dataset = dummyDataset(ch=3)
	sample = Dataset[0]
	target,model_input = sample['target'],np.expand_dims(sample['input'],0)
	N,C,H,W = model_input.shape
	net = MW_Unet(in_ch=C)
	output = net(torch.Tensor(model_input))
	output = output.squeeze(0).detach().numpy().transpose(2,1,0)
	output = (255 * (output-np.min(output))/np.ptp(output)).astype(int)
	print(output.dtype)
	#print(np.ptp(output,axis=2))

	#print(output[:,0,1:10,1:10])
	#fig = plt.figure()
	fig,ax  = plt.subplots(3)
	fig.subplots_adjust(hspace=0.5)
	ax[0].set_title('target')
	ax[0].imshow(target.transpose(2,1,0))
	ax[1].set_title('input')
	ax[1].imshow(model_input.squeeze(0).transpose(2,1,0))
	ax[2].set_title('output')
	ax[2].imshow(output)
	plt.show()



