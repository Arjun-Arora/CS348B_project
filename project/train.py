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
def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def create_args():
    parser = argparse.ArgumentParser(description = "hyperparameters for training")
    # parser.add_argument('--epochs',dest = 'epochs',default = 5, type=int,
    #                 help='number of epochs')
    parser.add_argument('--batch_size',dest = 'batch_size',default = 8, type=int,
                      help='size of batch')
    parser.add_argument('--lr',dest = 'lr',default = 0.01, type=float,
                      help='learning rate')
    parser.add_argument('--gpu',dest = 'gpu',default = False, action="store_true",
                        help='whether to use gpu')
    # parser.add_argument('--load_model',dest = 'load',default = False,
    #                   help='path to prev model')
    parser.add_argument('--val_split',dest = 'val_split',default = 0.2,type=float,
                         help = 'percentage of data in val')
    # parser.add_argument('--one_class',dest = 'one_class',default = False,action="store_true",
    #                   help='whether to make binary classification')
    # parser.add_argument('--non_background_weight',dest = 'non_background_weight',default = 40, type=float,
    #                 help='non_background_weight')
    # parser.add_argument('--network_rd_factor',dest = 'network_rd_factor',default = 0, type=int,
    #                 help='reduction factor of network rate')
    # parser.add_argument('--scale',dest = 'scale_factor',default = 0.1, type=float,
    #                 help='scale factor for training set')
    parser.add_argument('--print_every',dest='print_every',default=1,type=int,help='print every x epochs')
    parser.add_argument('--save_every',dest='save_every',default=1,type=int,help='save model every x epochs')
    parser.add_argument('--model_save_path',dest='model_save_path',default='./results/',type=str,help='path for saved model')
    parser.add_argument('--num_epochs',dest='num_epochs',default=10,type=int,help='number of epochs')
    parser.add_argument('--in_ch',dest='in_ch',default=9,type=int,help='number of channels')
    parser.add_argument('--image_directory',dest='image_directory',default='./results/',type=str,help='path for saved grid images')
    args = parser.parse_args()
    return args


def train(args,Dataset): 
    ####################################### Initializing Model #######################################
    step = args.lr
    #experiment_dir = args['--experiment_dir']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:{}".format(device))
    print_every = int(args.print_every)
    num_epochs = int(args.num_epochs)
    save_every = int(args.save_every)
    save_path = str(args.model_save_path)
    batch_size = int(args.batch_size)
    #train_data_path = str(args['--data_path'])
    in_ch = int(args.in_ch)
    val_split = args.val_split
    img_directory = args.image_directory
    #model = MW_Unet(in_ch=in_ch)
    model = UNet(in_ch=in_ch)
    #model = model
    model.to(device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=step)


    #criterion = nn.MSELoss()
    criterion = torch.nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        ######################################### Loading Data ##########################################

    dataset_total = Dataset
    dataset_size = len(dataset_total)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    #train_indices, val_indices = indices[:1], indices[1:2]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(dataset_total, batch_size=batch_size,sampler=train_sampler,num_workers=8)
    dataloader_val = torch.utils.data.DataLoader(dataset_total, batch_size=batch_size,sampler=valid_sampler,num_workers=2)

    print("length of train set: ", len(train_indices))
    print("length of val set: ", len(val_indices))

    #best_val_PSNR = 0.0
    best_val_MSE = 100.0 

    train_PSNRs = []
    train_losses = []
    train_SSIMs = []
    train_MSEs = []

    val_PSNRs = []
    val_losses = []
    val_SSIMs = []
    val_MSEs = []

    try:
        for epoch in range(1, num_epochs + 1):
                # INITIATE dataloader_train
                print("epoch: ", epoch)
                with tqdm(total=len(dataloader_train)) as pbar:
                        for index, sample in enumerate(dataloader_train):
                            model.train()

                            target, model_input, features = sample['target'],sample['input'], sample['features']
                            N,P,C,H,W = model_input.shape
                            N,P,C_feat,H,W = features.shape
                            model_input = torch.reshape(model_input,(-1,C,H,W))
                            features = torch.reshape(features,(-1,C_feat,H,W))
                            albedo = features[:,3:,:,:]
                            albedo = albedo.to(device)
                            eps = torch.tensor(1e-6)
                            eps = eps.to(device)
                            model_input = model_input.to(device)
                            model_input /= (albedo + eps)
                            target = torch.reshape(target,(-1,C,H,W))
                            features = features.to(device)
                            model_input = torch.cat((model_input, features), dim=1)
                            target = target.to(device)
                            model_input = model_input.to(device)

                            #print(model_input.dtype)
                            #print(model_input.shape)
                            # print(index)

                            output = model.forward(model_input)
                            output *= (albedo + eps)

                            train_loss = utils.backprop(optimizer, output, target,criterion)
                            train_PSNR = utils.get_PSNR(output, target)
                            train_MSE = utils.get_MSE(output,target)
                            train_SSIM = utils.get_SSIM(output,target)

                            avg_val_PSNR = []
                            avg_val_loss = []
                            avg_val_MSE = []
                            avg_val_SSIM =[]
                            model.eval()
                            #output_val = 0;

                            train_losses.append(train_loss.cpu().detach().numpy())
                            train_PSNRs.append(train_PSNR)
                            train_MSEs.append(train_MSE)
                            train_SSIMs.append(train_SSIM)

                            if index == len(dataloader_train)  - 1:
                                with torch.no_grad():
                                    for val_index, val_sample in enumerate(dataloader_val):
                                            target_val, model_input_val, features_val = val_sample['target'],val_sample['input'], val_sample['features']
                                            N,P,C,H,W = model_input_val.shape
                                            N,P,C_feat,H,W = features_val.shape
                                            model_input_val =torch.reshape(model_input_val,(-1,C,H,W))
                                            features_val = torch.reshape(features_val,(-1,C_feat,H,W))
                                            albedo = features_val[:,3:,:,:]
                                            albedo = albedo.to(device)
                                            eps = torch.tensor(1e-6)
                                            eps = eps.to(device)
                                            model_input_val = model_input_val.to(device)
                                            model_input_val /= (albedo + eps)
                                            target_val = torch.reshape(target_val,(-1,C,H,W))
                                            features_val = features_val.to(device)
                                            model_input_val = torch.cat((model_input_val, features_val), dim=1)
                                            target_val = target_val.to(device)
                                            model_input_val = model_input_val.to(device)
                                            output_val = model.forward(model_input_val)
                                            output_val *= (albedo + eps)
                                            loss_fn = criterion
                                            loss_val = loss_fn(output_val, target_val)
                                            PSNR = utils.get_PSNR(output_val, target_val)
                                            MSE = utils.get_MSE(output_val,target_val)
                                            SSIM = utils.get_SSIM(output_val,target_val)
                                            avg_val_PSNR.append(PSNR)
                                            avg_val_loss.append(loss_val.cpu().detach().numpy())
                                            avg_val_MSE.append(MSE)
                                            avg_val_SSIM.append(SSIM)

                                avg_val_PSNR = np.mean(avg_val_PSNR)
                                avg_val_loss = np.mean(avg_val_loss)
                                avg_val_MSE = np.mean(avg_val_MSE)
                                avg_val_SSIM = np.mean(avg_val_SSIM)

                                val_PSNRs.append(avg_val_PSNR)
                                val_losses.append(avg_val_loss)
                                val_MSEs.append(avg_val_MSE)
                                val_SSIMs.append(avg_val_SSIM)
                                scheduler.step(avg_val_loss)

                                img_grid = output.data[:9]
                                img_grid = torchvision.utils.make_grid(img_grid)
                                real_grid = target.data[:9]
                                real_grid = torchvision.utils.make_grid(real_grid)
                                input_grid = model_input.data[:9,:3,:,:]
                                input_grid = torchvision.utils.make_grid(input_grid)
                                val_grid = output_val.data[:9]
                                val_grid = torchvision.utils.make_grid(val_grid)
                                #save_image(input_grid, '{}train_input_img.png'.format(img_directory))
                                #save_image(img_grid, '{}train_img_{}.png'.format(img_directory, epoch))
                                #save_image(real_grid, '{}train_real_img_{}.png'.format(img_directory, epoch))
                                #print('train images')
                                fig,ax  = plt.subplots(4)
                                fig.subplots_adjust(hspace=0.5)
                                ax[0].set_title('target')
                                ax[0].imshow(real_grid.cpu().numpy().transpose((1, 2, 0)))
                                ax[1].set_title('input')
                                ax[1].imshow(input_grid.cpu().numpy().transpose((1, 2, 0)))
                                ax[2].set_title('output_train')
                                ax[2].imshow(img_grid.cpu().numpy().transpose((1, 2, 0)))
                                ax[3].set_title('output_val')
                                ax[3].imshow(val_grid.cpu().numpy().transpose((1, 2, 0)))
                                #plt.show()
                                plt.savefig('{}train_output_target_img_{}.png'.format(img_directory, epoch))
                                plt.close()

                            pbar.update(1)
                if epoch % print_every == 0:
                        print("Epoch: {}, Loss: {}, Train MSE: {} Train PSNR: {}, Train SSIM: {}".format(epoch, train_loss,train_MSE, train_PSNR,train_SSIM))
                        print("Epoch: {}, Avg Val Loss: {}, Avg Val MSE: {}, Avg Val PSNR: {}, Avg Val SSIM: {}".format(epoch, avg_val_loss,avg_val_MSE, avg_val_PSNR,avg_val_SSIM))
                        plt.figure()
                        plt.plot(np.linspace(0,epoch,len(train_losses)),train_losses)
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.savefig("{}train_loss.png".format(img_directory))
                        plt.close()

                        plt.figure()
                        plt.plot(np.linspace(0,epoch,len(val_losses)),val_losses)
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.savefig("{}val_loss.png".format(img_directory))
                        plt.close()

                        plt.figure()
                        plt.plot(np.linspace(0,epoch,len(train_PSNRs)),train_PSNRs)
                        plt.xlabel("Epoch")
                        plt.ylabel("PSNR")
                        plt.savefig("{}train_PSNR.png".format(img_directory))
                        plt.close()

                        plt.figure()
                        plt.plot(np.linspace(0,epoch,len(val_PSNRs)),val_PSNRs)
                        plt.xlabel("Epoch")
                        plt.ylabel("PSNR")
                        plt.savefig("{}val_PSNR.png".format(img_directory))
                        plt.close()

                        plt.figure()
                        plt.plot(np.linspace(0,epoch,len(train_MSEs)),train_MSEs)
                        plt.xlabel("Epoch")
                        plt.ylabel("MSE")
                        plt.savefig("{}train_MSE.png".format(img_directory))
                        plt.close()

                        plt.figure()
                        plt.plot(np.linspace(0,epoch,len(val_MSEs)),val_MSEs)
                        plt.xlabel("Epoch")
                        plt.ylabel("MSE")
                        plt.savefig("{}val_MSE.png".format(img_directory))
                        plt.close()


                        plt.figure()
                        plt.plot(np.linspace(0,epoch,len(train_SSIMs)),train_SSIMs)
                        plt.xlabel("Epoch")
                        plt.ylabel("SSIM")
                        plt.savefig("{}train_SSIM.png".format(img_directory))
                        plt.close()

                        plt.figure()
                        plt.plot(np.linspace(0,epoch,len(val_SSIMs)),val_SSIMs)
                        plt.xlabel("Epoch")
                        plt.ylabel("SSIM")
                        plt.savefig("{}val_SSIM.png".format(img_directory))
                        plt.close()
                        

                if best_val_MSE > avg_val_MSE:
                        best_val_MSE = avg_val_MSE
                        print("new best Avg Val MSE: {}".format(best_val_MSE))
                        print("Saving model to {}".format(save_path))
                        torch.save({'epoch': epoch,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': train_loss},
                                   save_path + "best_model.pth")
                        print("Saved successfully to {}".format(save_path))


    except KeyboardInterrupt:
            print("Training interupted...")
            print("Saving model to {}".format(save_path))
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss},
                       save_path + "checkpoint{}.pth".format(epoch))
            print("Saved successfully to {}".format(save_path))

            print("Training completed.")

    return (train_losses, train_PSNRs, val_losses, val_PSNRs, best_val_MSE)

if __name__ == "__main__":
    patchify = True
    dataset_dir0 = "./contemporary-bathroom_data/"
    dataset_dir1 = "./villa_data/"
    Dataset0 = utils.MonteCarloDataset(dataset_dir0,patchify=patchify)
    Dataset1 = utils.MonteCarloDataset(dataset_dir1,patchify=patchify)
    Dataset = torch.utils.data.ConcatDataset([Dataset0,Dataset1])
    args = create_args()
    train(args, Dataset)
