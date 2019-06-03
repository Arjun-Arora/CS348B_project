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
import argparse
import utils

def create_args():
    parser = argparse.ArgumentParser(description = "hyperparameters for training")
    # parser.add_argument('--epochs',dest = 'epochs',default = 5, type=int,
    #                 help='number of epochs')
    parser.add_argument('--batch_size',dest = 'batch_size',default = 16, type=int,
                      help='size of batch')
    parser.add_argument('--lr',dest = 'lr',default = 0.001, type=float,
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
    parser.add_argument('--model_save_path',dest='model_save_path',default='./',type=str,help='path for saved model')
    parser.add_argument('--num_epochs',dest='num_epochs',default=10,type=int,help='number of epochs')
    parser.add_argument('--in_ch',dest='in_ch',default=9,type=int,help='number of channels')
    parser.add_argument('--image_directory',dest='image_directory',default='./',type=str,help='path for saved grid images')
    args = parser.parse_args()
    return args


def train(args,Dataset): 
    ####################################### Initializing Model #######################################
    step = args.lr
    #experiment_dir = args['--experiment_dir']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_every = int(args.print_every)
    num_epochs = int(args.num_epochs)
    save_every = int(args.save_every)
    save_path = str(args.model_save_path)
    batch_size = int(args.batch_size)
    #train_data_path = str(args['--data_path'])
    in_ch = int(args.in_ch)
    val_split = args.val_split
    img_directory = args.img_directory

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

                            if index == len(dataloader_train) - 1:
                                    img_grid = output.data
                                    img_grid = torchvision.utils.make_grid(img_grid)
                                    real_grid = target.data
                                    real_grid = torchvision.utils.make_grid(real_grid)
                                    input_grid = model_input.data
                                    input_grid = torchvision.utils.make_grid(input_grid)
                                    #save_image(input_grid, '{}train_input_img.png'.format(img_directory))
                                    #save_image(img_grid, '{}train_img_{}.png'.format(img_directory, epoch))
                                    #save_image(real_grid, '{}train_real_img_{}.png'.format(img_directory, epoch))
                                    print('train images')
                                    fig,ax  = plt.subplots(3)
                                    fig.subplots_adjust(hspace=0.5)
                                    ax[0].set_title('target')
                                    ax[0].imshow(real_grid.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
                                    ax[1].set_title('input')
                                    ax[1].imshow(input_grid.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
                                    ax[2].set_title('output')
                                    ax[2].imshow(img_grid.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
                                    #plt.show()
                                    plt.savefig('{}train_output_target_img_{}.png'.format(img_directory, epoch))

                            pbar.update(1)
                            if epoch % print_every == 0:
                                    print("Epoch: {}, Loss: {}, Training PSNR: {}".format(epoch, train_loss, train_PSNR))
                                    print("Epoch: {}, Avg Val Loss: {},Avg Val PSNR: {}".format(epoch, avg_val_loss, avg_val_PSNR))
                if best_val_PSNR < avg_val_PSNR:
                        best_val_PSNR = avg_val_PSNR
                        print("new best Avg Val PSNR: {}".format(best_val_PSNR))
                        print("Saving model to {}".format(save_path))
                        torch.save({'epoch': epoch,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': train_loss},
                                   save_path)
                        print("Saved successfully to {}".format(save_path))


    except KeyboardInterrupt:
            print("Training interupted...")
            print("Saving model to {}".format(save_path))
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss},
                       save_path)
            print("Saved successfully to {}".format(save_path))

            print("Training completed.")

    return (train_losses, train_PSNRs, val_losses, val_PSNRs, best_val_PSNR)

if __name__ == "__main__":
    patchify = True
    dataset_dir = "./contemporary-bathroom_data/"
    Dataset = utils.MonteCarloDataset(dataset_dir,patchify=patchify)
    args = create_args()
    train(args, Dataset)
