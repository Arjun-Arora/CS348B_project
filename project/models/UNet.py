import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Flatten(nn.Module): 
	def forward(self,x): 
		return x.view(x.shape[0],-1)

class CONV_FINAL(nn.Module):
	"""
	CONV -> BN -> RELU layer
	inputs: 
	in_ch: input channels
	out_ch: output channels
	filter_sz: size of filters
	NOTE: Padding and stride are not input since we will use same padding.
	Unet architectures are particular about sizes 
	"""
	def __init__(self,in_ch,out_ch,filter_sz = 3):
		super(CONV_FINAL,self).__init__()

		pad = int((-1 + filter_sz)/2)
		self.forward_prop = nn.Sequential(
			nn.Conv2d(in_ch,out_ch,filter_sz,padding = pad),
			nn.Conv2d(out_ch,out_ch,filter_sz,padding = pad)
		)
	def forward(self,x):
		x = self.forward_prop(x)
		return x 	


class CONV_BN_RELU(nn.Module):
	"""
	CONV -> BN -> RELU layer
	inputs: 
	in_ch: input channels
	out_ch: output channels
	filter_sz: size of filters
	NOTE: Padding and stride are not input since we will use same padding.
	Unet architectures are particular about sizes 
	"""
	def __init__(self,in_ch,out_ch,filter_sz = 3):
		super(CONV_BN_RELU,self).__init__()

		pad = int((-1 + filter_sz)/2)
		self.forward_prop = nn.Sequential(
			nn.Conv2d(in_ch,out_ch,filter_sz,padding = pad),
			nn.BatchNorm2d(out_ch),
			nn.ReLU()
			)
	def forward(self,x):
		x = self.forward_prop(x)
		return x 

class upConv(nn.Module):
	"""
	ConvTranpose2d -> CONV_BN_RELU layer
	inputs: 
	in_ch: input channesl
	out_ch output channels 
	filter_sz: size of filter (must be > 2)
	conv_filter_sz: size of filter for the standard convolution
	This layer increases H and W by factor of 2 
	"""
	def __init__(self,in_ch,out_ch,filter_sz = 2,conv_filter_sz = 3,bilinear=True):
		super(upConv,self).__init__()

		if(filter_sz <  2): 
			raise Exception('filter_sz must be atleast 2. The value of filter_sz was: {}'.format(filter_sz))
		padding = int((-2 + filter_sz)/2)
		if bilinear:
			self.upConv = nn.Upsample(scale_factor = 2, mode = 'bilinear',align_corners=True)
		else:
			self.upConv = nn.ConvTranspose2d(int(in_ch/2) , int(in_ch/2),filter_sz,2,padding)

		self.conv = nn.Sequential(
			CONV_BN_RELU(in_ch,out_ch,conv_filter_sz),
			CONV_BN_RELU(out_ch,out_ch,conv_filter_sz),
			)

	def forward(self,exp_input,cont_input):
		"""
		exp_input: from the expanding network (needs to have upConv applied)
		cont_input: from the contracting network (at the size needed)
		NOTE: inputs are assumed to be in standard pytorch format: N,C,H,W
		""" 
		exp_input = self.upConv(exp_input);

		#calculate padding required to ensure we can stack exp_input and cont_input

		H_dim_diff = cont_input.shape[2] - exp_input.shape[2]
		W_dim_diff = cont_input.shape[3] - exp_input.shape[3]

		#padding exp_input to match cont_input's shape
		exp_input = F.pad(exp_input, ( int(W_dim_diff/2) , W_dim_diff - int(W_dim_diff/2), 
									   int(H_dim_diff/2) , H_dim_diff - int(H_dim_diff/2)  ) )

		combined = torch.cat([cont_input,exp_input], dim = 1 )

		combined = self.conv(combined)

		return combined

class downConv(nn.Module): 
	"""
	Downsample -> Conv layer 
	inputs:
	in_ch: input channesl
	out_ch: output channels 
	filter_sz: size of filter of ocnv layer
	"""
	def __init__(self,in_ch,out_ch,filter_sz = 3):
		super(downConv,self).__init__()	
		self.conv = nn.Sequential(
			CONV_BN_RELU(in_ch,out_ch,filter_sz),
			CONV_BN_RELU(out_ch,out_ch,filter_sz),
			)
		self.pool = nn.MaxPool2d(2,2)
	def forward(self,x):
		# x = F.interpolate(x,scale_factor = 0.5)
		x = self.pool(x)
		return self.conv(x)

class UNet(nn.Module):
	"""
	UNet 
	in_ch: number of input channels to UNet 
	out_ch: number of output channels for the Unet 
	"""
	def __init__(self,in_ch = 3,out_ch = 3):
		super().__init__()
		layer_dim_1 = 16 
		layer_dim_2 = layer_dim_1 * 2
		layer_dim_3 = layer_dim_2 * 2
		layer_dim_4 = layer_dim_3 * 2
		layer_dim_5 = layer_dim_4 * 2

		self.in_conv = nn.Sequential(
			CONV_BN_RELU(in_ch,layer_dim_1),
			CONV_BN_RELU(layer_dim_1,layer_dim_1))

		self.down1 = downConv(layer_dim_1,layer_dim_2)
		self.down2 = downConv(layer_dim_2,layer_dim_3)
		self.down3 = downConv(layer_dim_3,layer_dim_4)
		self.down4 = downConv(layer_dim_4,layer_dim_4)
		self.up1 = upConv(layer_dim_5,layer_dim_3)
		self.up2 = upConv(layer_dim_4,layer_dim_2)
		self.up3 = upConv(layer_dim_3,layer_dim_1)
		self.up4 = upConv(layer_dim_2,layer_dim_1)
		# self.down1 = downConv(layer_dim_1,layer_dim_2)
		# self.down2 = downConv(layer_dim_2,layer_dim_3)
		# self.down3 = downConv(layer_dim_3,layer_dim_3)
		# self.up1 = upConv(layer_dim_4,layer_dim_2)
		# self.up2 = upConv(layer_dim_3,layer_dim_1)
		# self.up3 = upConv(layer_dim_2,layer_dim_1)
		self.out_conv = nn.Sequential(
			CONV_BN_RELU(layer_dim_1,layer_dim_1),
			CONV_FINAL(layer_dim_1,out_ch))

	def forward(self,x): 
		# layer1 = self.in_conv(x)
		# layer2 = self.down1(layer1)
		# layer3 = self.down2(layer2)
		# layer4 = self.down3(layer3)
		# output = self.up1(layer4,layer3)
		# output = self.up2(output,layer2)
		# output = self.up3(output,layer1)
		# output = self.out_conv(output)
		layer1 = self.in_conv(x)
		layer2 = self.down1(layer1)
		layer3 = self.down2(layer2)
		layer4 = self.down3(layer3)
		layer5 = self.down4(layer4)
		output = self.up1(layer5,layer4)
		output = self.up2(output,layer3)
		output = self.up3(output,layer2)
		output = self.up4(output,layer1)
		output = self.out_conv(output)
		return output

if __name__ == "__main__":
	print("testing UNet")
	#N,C,H,W
	X = torch.randn(10,9,128,128)
	print("X shape: {} ".format(X.shape))
	N,C,H,W = X.shape
	Y = torch.randn(10,3,128,128)
	print("Y shape: {}".format(Y.shape))
	Unet = UNet(C)
	Output = Unet(X)
	print("Output shape: {}".format(Output.shape))