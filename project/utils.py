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

    def __init__(self,data_path,patchify=True,patch_sz=(128,128),max_patches = 8):
        self.data_path = data_path
        self.patchify = patchify
        self.image_list,self.feature_list,self.target_list = loadFilenames(data_path)
        #print(len(self.image_list))
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
            depth_map = feature_map[:,:,:,2]
            feature_map[:,:,:,3:] = feature_map[:,:,:,3:]**(.2)
            depth_map = depth_map/np.max(depth_map)
            target_img = np.transpose(target_img,(0,3,1,2))
            input_img = np.transpose(input_img,(0,3,1,2))
            feature_map = np.transpose(feature_map,(0,3,1,2))
        else:
            depth_map = feature_map[:,:,2]
            feature_map[:,:,3:] = feature_map[:,:,3:]**(.2)
            depth_map = depth_map/np.max(depth_map)
            target_img = np.transpose(target_img,(2,0,1))
            input_img = np.transpose(input_img,(2,0,1))
            feature_map = np.transpose(feature_map,(2,0,1))

        input_img = input_img**(.2)
        target_img = target_img**(.2)
        

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
def get_MSE(model_output,target):
    I_hat = model_output.cpu().detach().numpy()
    I = target.cpu().detach().numpy()
    mse = (np.square(I - I_hat)).mean(axis=None)
    return mse 


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

if __name__ =="__main__":
    patchify = False
    #dataset_dir = "./contemporary-bathroom_data/"
    dataset_dir = "./villa_data"
    #dataset_dir = './breakfast-sample'
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



