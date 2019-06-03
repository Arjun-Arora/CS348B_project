import numpy as np 
import torch
import csv
import skimage as sk
import os 
import sys
import glob
from sklearn.feature_extraction import image
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import torchvision
import matplotlib.pyplot as plt 
import torch.nn as nn
from skimage.measure import compare_ssim as ssim
sys.path.append("./models/")
from MWU_CNN import MW_Unet 
from tqdm import tqdm 
import glob, os 
import fnmatch
import cv2


def loadFilenames(data_path):
	image_list = []
	feature_list = []
	target_list = []
	for file in os.listdir(data_path):
		if file.endswith("_64.exr"):
			image_list.append(file)
		if file.endswith("_64.npy"):
			feature_list.append(file)
		if file.endswith("_4096.exr"):
			target_list.append(file)
	return sorted(image_list),sorted(feature_list),sorted(target_list)


class MonteCarloDataset(Dataset):

	def __init__(self,data_path,patchify=True,patch_sz=(64,64),max_patches = 32):
		self.data_path = data_path
		self.patchify = patchify
		self.image_list,self.feature_list,self.target_list = loadFilenames(data_path)

		assert len(self.image_list) == len(self.target_list) == len(self.feature_list)

		self.length = len(self.image_list)
		self.patch_sz = patch_sz
		self.max_patches = max_patches
	def __len__(self):
		return self.length
	def __getitem__(self,idx): 
		input_img = cv2.imread(os.path.join(self.data_path,
							 self.image_list[idx]),
							 cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
		input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
		#print(input_img.shape)
		#print(input_img.dtype)
		feature_map = np.load(os.path.join(self.data_path,self.feature_list[idx]))
		target_img = cv2.imread(os.path.join(self.data_path,self.target_list[idx]),cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
		target_img = cv2.cvtColor(target_img,cv2.COLOR_BGR2RGB)

		if self.patchify:
			stacked_img = np.dstack((input_img,target_img,feature_map))
			patched = image.extract_patches_2d(stacked_img,self.patch_sz,max_patches=self.max_patches)
			#print(patched.shape)
			input_img = patched[:,:,:,0:3]
			target_img = patched[:,:,:,3:6]
			feature_map = patched[:,:,:,6:]
			depth_map = feature_map[:,:,2]
			depth_map = depth_map/np.max(depth_map)
			target_img = np.transpose(target_img,(0,3,1,2))
			input_img = np.transpose(input_img,(0,3,1,2))
			feature_map = np.transpose(feature_map,(0,3,1,2))
		else:
			depth_map = feature_map[:,:,2]
			depth_map = depth_map/np.max(depth_map)
			target_img = np.transpose(target_img,(2,0,1))
			input_img = np.transpose(input_img,(2,0,1))
			feature_map = np.transpose(feature_map,(2,0,1))

		sample = {'input':input_img,'features': feature_map,'target':target_img}
		return sample



class dummyDataset(Dataset):
	'''
	This is a dummy dataset that just returns tensors of img_sz with ch channels
	Output format is: N,C,H,W 
	'''
	def __init__(self,ch=5,img_sz = (16,16),length = 10000):
		assert len(img_sz) == 2
		self.ch = ch
		self.img_sz = img_sz
		self.length = length
		self.sz = (self.ch,self.img_sz[0],self.img_sz[1])
	def __len__(self):
		return self.length
	def __getitem__(self,idx):
		input_img = np.random.randn(*self.sz)
		target_img = np.random.randn(*self.sz)
		sample = {'target':target_img,'input':input_img}
		return sample
def backprop(optimizer, model_output, target,criterion):
		optimizer.zero_grad()
		loss_fn = criterion
		loss = loss_fn(model_output, target)
		loss.backward()
		optimizer.step()
		return loss


def get_PSNR(model_output, target):
		I_hat = model_output.cpu().detach().numpy()
		I = target.cpu().detach().numpy()
		mse = (np.square(I - I_hat)).mean(axis=None)
		PSNR = 10 * np.log10(1.0 / mse)
		return PSNR


def get_SSIM(model_output, target):
		I_hat = model_output.cpu().detach().numpy()
		I = target.cpu().detach().numpy()
		N, C, H, W = I_hat.shape
		ssim_out = []
		for i in range(N):
				img = I[i, 0, :, :]
				img_noisy = I_hat[i, 0, :, :]
				ssim_out.append(ssim(img, img_noisy, data_range=img_noisy.max() - img_noisy.min()))
		return np.mean(ssim_out)

# def imshow(img):
#     #     print('Image device and mean')
#     #     print(img.device)
#     #     print(img.mean())
#     output_image = img.cpu().numpy().transpose((1, 2, 0))
#     npimg = output_image.astype(np.uint8)
#     #     print('Mean of image: {}'.format(npimg.mean()))
#     # format H,W,C
#     plt.imshow(npimg)
#     plt.show()

def train(args,Dataset): 
	####################################### Initializing Model #######################################
	step = 0.01
	experiment_dir = args['--experiment_dir']
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print_every = int(args['--print_every'])
	num_epochs = int(args['--num_epochs'])
	save_every = int(args['--save_every'])
	save_path = str(args['--model_save_path'])
	batch_size = int(args['--batch_size'])
	train_data_path = str(args['--data_path'])
	in_ch = int(args['--in_ch'])
	val_split = args['--val_split']

	model = MW_Unet(in_ch=in_ch)
	model = model.double()
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=step)

	criterion = nn.MSELoss()

		######################################### Loading Data ##########################################

	dataset_total = Dataset
	dataset_size = len(dataset_total)
	indices = list(range(dataset_size))
	split = int(np.floor(val_split * dataset_size))
	np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	dataloader_train = torch.utils.data.DataLoader(dataset_total, batch_size=batch_size, 
																				 sampler=train_sampler)
	dataloader_val = torch.utils.data.DataLoader(dataset_total, batch_size=batch_size,
																							sampler=valid_sampler)

	print("length of train set: ", len(train_indices))
	print("length of val set: ", len(val_indices))

	best_val_PSNR = 0.0
	train_PSNRs = []
	train_losses = []
	val_PSNRs = []
	val_losses = []

	try:
		for epoch in range(1, num_epochs + 1):
				# INITIATE dataloader_train
				print("epoch: ", epoch)
				with tqdm(total=len(dataloader_train)) as pbar:
						for index, sample in enumerate(dataloader_train):
							model.train()

							target, model_input = sample['target'],sample['input']
							target = target.to(device)
							model_input = model_input.to(device)

							print(model_input.dtype)
							print(model_input.shape)
							# print(index)

							output = model.forward(model_input.double())

							train_loss = backprop(optimizer, output, target,criterion)
							train_PSNR = get_PSNR(output, target)

							avg_val_PSNR = []
							avg_val_loss = []
							model.eval()
							with torch.no_grad():
									for val_index, val_sample in enumerate(dataloader_val):
											target, model_input = val_sample['target'], val_sample['input']

											target = target.to(device)
											model_input = model_input.to(device)

											output = model.forward(model_input)
											loss_fn = criterion
											loss_val = loss_fn(output, target)
											PSNR = get_PSNR(output, target)
											avg_val_PSNR.append(PSNR)
											avg_val_loss.append(loss_val.cpu().detach().numpy())
							avg_val_PSNR = np.mean(avg_val_PSNR)
							avg_val_loss = np.mean(avg_val_loss)
							val_PSNRs.append(avg_val_PSNR)
							val_losses.append(avg_val_loss)

							train_losses.append(train_loss.cpu().detach().numpy())
							train_PSNRs.append(train_PSNR)

							if True:#index == len(dataloader_train) - 1:
									img_grid = output.data
									img_grid = torchvision.utils.make_grid(img_grid)
									real_grid = target.data
									real_grid = torchvision.utils.make_grid(real_grid)
									input_grid = model_input.data
									input_grid = torchvision.utils.make_grid(input_grid)
									#directory = img_directory
									# save_image(input_grid, '{}train_input_img.png'.format(directory))
									# save_image(img_grid, '{}train_img_{}.png'.format(directory, epoch))
									# save_image(real_grid, '{}train_real_img_{}.png'.format(directory, epoch))
									#print('train images')
									fig,ax  = plt.subplots(3)
									fig.subplots_adjust(hspace=0.5)
									ax[0].set_title('target')
									ax[0].imshow(real_grid.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
									ax[1].set_title('input')
									ax[1].imshow(input_grid.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
									ax[2].set_title('output')
									ax[2].imshow(img_grid.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
									plt.show()

							pbar.update(1)
							if epoch % print_every == 0:
									print("Epoch: {}, Loss: {}, Training PSNR: {}".format(epoch, train_loss, train_PSNR))
									print("Epoch: {}, Avg Val Loss: {},Avg Val PSNR: {}".format(epoch, avg_val_loss, avg_val_PSNR))
					# if epoch % save_every == 0 and best_val_PSNR < avg_val_PSNR:
					#     best_val_PSNR = avg_val_PSNR
					#     print("new best Avg Val PSNR: {}".format(best_val_PSNR))
					#     print("Saving model to {}".format(save_path))
					#     torch.save({'epoch': epoch,
					#                 'model_state_dict': model.state_dict(),
					#                 'optimizer_state_dict': optimizer.state_dict(),
					#                 'loss': train_loss},
					#                save_path)
						#     print("Saved successfully to {}".format(save_path))


	except KeyboardInterrupt:
			# print("Training interupted...")
			# print("Saving model to {}".format(save_path))
			# torch.save({'epoch': epoch,
			#             'model_state_dict': model.state_dict(),
			#             'optimizer_state_dict': optimizer.state_dict(),
			#             'loss': train_loss},
			#            save_path)
			# print("Saved successfully to {}".format(save_path))

		print("Training completed.")

	return (train_losses, train_PSNRs, val_losses, val_PSNRs, best_val_PSNR)





if __name__ =="__main__":
	patchify = False
	dataset_dir = "./contemporary-bathroom_data/"
	Dataset = MonteCarloDataset(dataset_dir,patchify=patchify)
	idx = np.random.randint(0,len(Dataset))
	sample = Dataset[idx]
	#output in C,H,W format
	if patchify:
		input_patches,feature_map_patches,target_patches = sample['input'],sample['features'],sample['target']
		N,C,H,W = input_patches.shape
		patch_idx = np.random.randint(0,N)
		input_img = input_patches[patch_idx,:,:,:]
		feature_map = feature_map_patches[patch_idx,:,:,:]
		target_img = target_patches[patch_idx,:,:,:]
	else:
		input_img,feature_map,target_img = sample['input'],sample['features'],sample['target']

	depth_map = feature_map[2,:,:]
	normals = np.vstack((feature_map[0:2,:,:],np.expand_dims(np.zeros_like(depth_map),axis=0)))
	albedo = feature_map[3:,:,:]

	#print(feature_map.dtype)
	f,ax = plt.subplots(3,2)
	f.subplots_adjust(hspace=1.0)
	#f.set_size_inches(18.5,10.5)
	ax[0,0].imshow(np.transpose(input_img ** (1/2.2),(1,2,0)))
	ax[0,0].set_title("input")
	ax[0,1].imshow(np.transpose(target_img ** (1/2.2),(1,2,0)))
	ax[0,1].set_title("target")
	ax[1,0].imshow(depth_map,cmap="gray")
	ax[1,0].set_title('depth_map')
	ax[1,1].imshow(np.transpose(normals,(1,2,0)))
	ax[1,1].set_title("normal_map")
	ax[2,0].imshow(np.transpose(albedo,(1,2,0)))
	ax[2,0].set_title("albedo")
	f.delaxes(ax[2,1])
	plt.show()



	# print(len(image_list))
	# print(len(feature_list))
	# image_str = fnmatch.filter(image_list,idx_pattern_image)
	# feature_str = fnmatch.filter(feature_list,idx_pattern_feature)
	# print(image_str)
	# print(feature_str)
	# in_ch = 3 
	# Dataset = dummyDataset(ch=in_ch)


	# args = {}
	# args['--experiment_dir'] = "./" 
	# args['--print_every'] = 1
	# args['--num_epochs'] = 3 
	# args['--save_every'] = 1 
	# args['--model_save_path'] = "./"
	# args['--batch_size'] = 16
	# args['--val_split']  = 0.2
	# args['--in_ch'] = in_ch
	# args['--data_path'] = "./"
	# train(args,Dataset)



