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
from MWU_CNN import SimpleCNN
from UNet import UNet
from tqdm import tqdm 
import glob, os 
import fnmatch
import cv2
import argparse
import utils
import sys

model_path = sys.argv[1]
image_dir = sys.argv[2]
image_name = sys.argv[3]

def load_saved_model(model, map_location):
    checkpoint = torch.load(model_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])

def evaluate(model, device, model_input, features, target):
    model = model.to(device)
    og_input = np.copy(model_input)
    with torch.no_grad():
        model_input = torch.tensor(model_input)
        features = torch.tensor(features)
        target = torch.tensor(target)
        C, H, W = model_input.shape
        model_input = torch.reshape(model_input, (1, C, H, W))
        C_feat, H, W = features.shape
        features = torch.reshape(features, (1, C_feat, H, W))

        albedo = features[:, 3:, :, :]
        albedo = albedo.to(device)
        eps = torch.tensor(1e-6)
        eps = eps.to(device)
        model_input = model_input.to(device)
        model_input /= (albedo + eps)
        features = features.to(device)
        model_input = torch.cat((model_input, features), dim=1)
        model_input = model_input.to(device)
        output = model.forward(model_input)
        output *= (albedo + eps)

        C, H, W = target.shape
        target = torch.reshape(target, (1, C, H, W))
        target = target.to(device)
        PSNR = utils.get_PSNR(output, target)
        MSE = utils.get_MSE(output, target)
        SSIM = utils.get_SSIM(output, target)
        print("PSNR: %.10f, MSE: %.10f, SSIM: %.10f" % (PSNR, MSE, SSIM))

        plt.imsave(image_name + "_denoised.png", np.transpose(np.squeeze(output.cpu().numpy(), axis=0), (1, 2, 0)))
        '''
        fig,ax  = plt.subplots(2, 2)
        fig.subplots_adjust(hspace=0.5)
        ax[0, 0].set_title('target')
        ax[0, 0].imshow(np.transpose(np.squeeze(target.cpu().numpy(), axis=0), (1, 2, 0)))
        ax[0, 1].set_title('input')
        ax[0, 1].imshow(np.transpose(og_input, (1, 2, 0)))
        ax[1, 0].set_title('output')
        ax[1, 0].imshow(np.transpose(np.squeeze(output.cpu().numpy(), axis=0), (1, 2, 0)))
        fig.delaxes(ax[1, 1])
        plt.savefig("denoised_image.png")
        plt.figure(100)
        plt.imshow(np.transpose(np.squeeze(target.cpu().numpy(), axis=0), (1, 2, 0)))
        plt.savefig("target.png")
        '''

if __name__ == "__main__":
    model = UNet(in_ch=9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:{}".format(device))
    load_saved_model(model, map_location=device)

    input_img = cv2.imread(image_dir + image_name + "_64.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    feature_map = np.load(image_dir + "feature_map_" + image_name[-1] + "_64.npy")
    target_img = cv2.imread(image_dir + image_name + "_4096.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    depth_map = feature_map[:, :, 2]
    depth_map = depth_map/np.max(depth_map)
    target_img = np.transpose(target_img,(2, 0, 1))
    input_img = np.transpose(input_img,(2, 0, 1))
    feature_map = np.transpose(feature_map,(2, 0, 1))

    input_img = input_img**(1/2.2)
    target_img = target_img**(1/2.2)

    evaluate(model, device, input_img, feature_map, target_img)
