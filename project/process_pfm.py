import numpy as np
import re
import sys
import cv2
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import imageio

def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip().decode("utf-8")
    if header == 'PF':
      color = True    
    elif header == 'Pf':
      color = False
    else:
      raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode("utf-8"))
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape), scale

def save_pfm(file, image, scale = 1):
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)  

def get_PSNR(model_output, target):
    I_hat = model_output
    I = target
    mse = (np.square(I - I_hat)).mean(axis=None)
    PSNR = 10 * np.log10(1.0 / mse)
    return PSNR

def get_MSE(model_output,target):
    I_hat = model_output
    I = target
    mse = (np.square(I - I_hat)).mean(axis=None)
    return mse 

def get_SSIM(model_output, target):
    I_hat = model_output
    I = target
    N, C, H, W = I_hat.shape
    ssim_out = []
    for i in range(N):
        img = I[i, 0, :, :]
        img_noisy = I_hat[i, 0, :, :]
        ssim_out.append(ssim(img, img_noisy, data_range=img_noisy.max() - img_noisy.min(), multichannel=True))
    return np.mean(ssim_out)

NAME = sys.argv[1]

features = np.load(NAME + "/feature_map_0_64.npy")
H, W, _ = features.shape
normals = np.dstack((features[:, :, :2], np.zeros((H, W, 1), dtype=np.float32)))
depth = features[:, :, 2]
albedo = features[:, :, 3:]
save_pfm(open(NAME + "/normals.pfm", "w"), normals)
save_pfm(open(NAME + "/albedo.pfm", "w"), albedo)
plt.imsave(NAME + "/normals.png", normals)
plt.imsave(NAME + "/depth.png", depth, cmap="gray")
plt.imsave(NAME + "/albedo.png", albedo)

input_img = cv2.imread(NAME + "/" + NAME + "_0_64.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
save_pfm(open(NAME + "/input.pfm", "w"), input_img)
target_img = cv2.imread(NAME + "/" + NAME + "_0_4096.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
save_pfm(open(NAME + "/target.pfm", "w"), target_img)

denoised = imageio.imread(NAME + "/" + NAME + "_0_denoised.png")
plt.imsave(NAME + "/input_patch1.png", input_img[250:450, 200:720, :]**(1/2.2))
plt.imsave(NAME + "/target_patch1.png", target_img[250:450, 200:720, :]**(1/2.2))
plt.imsave(NAME + "/denoised_patch1.png", denoised[250:450, 200:720, :])
plt.imsave(NAME + "/input_patch2.png", input_img[280:530, 720:970, :]**(1/2.2))
plt.imsave(NAME + "/target_patch2.png", target_img[280:530, 720:970, :]**(1/2.2))
plt.imsave(NAME + "/denoised_patch2.png", denoised[280:530, 720:970, :])


# denoised, _ = load_pfm(open(NAME + "/intel_denoised.pfm", "rb"))
# plt.imsave(NAME + "/denoised.png", denoised**(1/2.2))

# target_img = np.expand_dims(target_img, axis=0)
# denoised = np.expand_dims(denoised, axis=0)
# print("PSNR: %.10f, MSE: %.10f, SSIM: %.10f" % (get_PSNR(denoised, target_img), get_MSE(denoised, target_img), get_SSIM(denoised, target_img)))