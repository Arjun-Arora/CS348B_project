import torch
import os
import sys
import torch.nn as nn
from MWCNN import WCNN,IWCNN
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self,in_ch=1,out_ch=3):
        super(SimpleCNN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 =torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 =torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        #self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #4608 input features, 64 output features (see sizing flow below)        
        #64 input features, 10 output features for our 10 defined classes
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.conv2(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.conv3(x)
        return(x)

class MW_Unet(nn.Module):
    """
    Baseline architecture for Multi-level Wavelet-CNN paper
    Incorporates Unet style concatenation of dims
    input:N,C,H,W
    output: N,C,H,W
    """
    def __init__(self,num_conv=0,in_ch=1,out_ch=3,channel_1=4,channel_2=8):
        '''
        :param: num_conv per contraction and expansion layer, how many extra conv-batch-relu layers wanted
        :param in_ch: number of input channels expected
        :return:
        '''
        super(MW_Unet,self).__init__()
        print("channel_1: {}, channel_2: {} num_conv: {}".format(channel_1,channel_2,num_conv))
        self.num_conv = num_conv
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.cnn_1 = WCNN(in_ch=in_ch,out_ch=channel_1,num_conv=num_conv) #output N,160,H/2,W/2
        self.cnn_2 = WCNN(in_ch=channel_1,out_ch=channel_2,num_conv=num_conv)
        self.cnn_3 = WCNN(in_ch=channel_2,out_ch=channel_2,num_conv=num_conv)
        self.icnn_3 = IWCNN(in_ch=channel_2,internal_ch=4*channel_2,num_conv=num_conv)
        self.icnn_2 = IWCNN(in_ch=2*channel_2,internal_ch=4*channel_1,num_conv=num_conv) #expecting 2*256 because of skip connection
        self.icnn_1 = IWCNN(in_ch=2*channel_1,internal_ch=self.in_ch*4,num_conv=num_conv) # output N,in_ch,H,W
        self.final_conv = nn.Conv2d(in_channels=self.in_ch,out_channels=self.out_ch,kernel_size=3,padding=1)

    def forward(self,x):
        x1 = self.cnn_1(x)
        x2 = self.cnn_2(x1)
        x3 = self.cnn_3(x2)

        y1 = self.icnn_3(x3)
        y2 = self.icnn_2(torch.cat((y1,x2),dim=1))
        y3 = self.icnn_1(torch.cat((y2,x1),dim=1))
        output = self.final_conv(y3)
        return output
if __name__ == "__main__":
    print("testing MW_Unet")
    X = torch.randn(10, 5, 64, 64)
    print(X.dtype)
    N, C, H, W = X.shape

    Unet = MW_Unet(in_ch=C)
    Unet.apply(init_weights)
    Y = Unet(X)


    print("shape of X: ", X.shape)
    print("shape of Y: ", Y.shape)
    #print(torch.mean(X - Y))
