import copy
import os
from PIL import Image
import numpy as np
import torch
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
import scipy.ndimage as ndimage
import skimage.measure as measure


# To make directories
def mkdir(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = np.mean(act1), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_fid2(act1, act2, eps=1e-6):
    # calculate mean and covariance statistics
    mu1, sigma1 = np.mean(act1), np.cov(act1, rowvar=True)
    mu2, sigma2 = np.mean(act2), np.cov(act2, rowvar=True)
    # diff = mu1 - mu2
    diff = np.sum((mu1 - mu2) ** 2.0)
    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            print('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


# To make cuda tensor
def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


# For Pytorch data loader
def create_link(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
    dirs['trainB'] = os.path.join(dataset_dir, 'ltrainB')
    dirs['testA'] = os.path.join(dataset_dir, 'ltestA')
    dirs['testB'] = os.path.join(dataset_dir, 'ltestB')
    mkdir(dirs.values())

    for key in dirs:
        try:
            os.remove(os.path.join(dirs[key], 'Link'))
        except:
            pass
        os.symlink(os.path.abspath(os.path.join(dataset_dir, key)),
                   os.path.join(dirs[key], 'Link'))

    return dirs


def np_to_img(image, mode):
    # y = I.detach().cpu().numpy()
    # image = y[0, 0, :, :]
    import sys
    ee = sys.float_info.epsilon
    I8 = (((image - image.min()) / (image.max() - image.min() + ee)) * 255.9).astype(np.uint8)
    img = Image.fromarray(I8, mode)
    return img


def get_traindata_link(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'ltrainA')
    dirs['trainB'] = os.path.join(dataset_dir, 'ltrainB')
    return dirs


def get_testdata_link(dataset_dir):
    dirs = {}
    dirs['testA'] = os.path.join(dataset_dir, 'ltestA')
    dirs['testB'] = os.path.join(dataset_dir, 'ltestB')
    return dirs


# To save the checkpoint 
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# saving images
def savingimage(tensor):
    y = tensor.detach().cpu().numpy()
    image = y[0, :, :]
    image = np.clip(image, -1.0, 1.0)
    image = (image + 1.0) / 2.0
    image = (image * 255.0).astype(np.uint8)
    img = Image.fromarray(image, 'L')
    return img
    
def savingimage2(tensor):
    y = tensor.detach().cpu().numpy()
    image = y[0, 0, :, :]
    image = np.clip(image, -0.46, 4.75)
    image = (image + 0.46) / 5.21
    image = (image * 255.0).astype(np.uint8)
    img = Image.fromarray(image, 'L')
    return img


# To load the checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


# To store 50 generated image in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch) / (self.epochs - self.decay_epoch)


def print_networks(nets, names):
    print('------------Number of Parameters---------------')
    i = 0
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))
        i = i + 1
    print('-----------------------------------------------')


def gradient_penalty(netD, real_data, fake_data, device='cuda:0', type='mixed', constant=1.0, lambda_gp=10.0):
    """
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty  # , gradients
    else:
        return 0.0  # , None


def activation_map(x, y):
    yy = y.detach().cpu().numpy()
    xx = x.detach().cpu().numpy()
    yyy = yy[0, :, :]
    xxx = xx[0, :, :]
    difference = yyy-xxx # abs(xxx - yyy)
    massimo = 2.0
    image = abs(xxx - yyy)
    image = image / massimo # considero come min = 0
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image, 'L')
    return image, difference


def keep_largest_mask(image_tensor):
    mask = image_tensor.clone()
    for i in range(10):
        image_tensor_2D = image_tensor[i, 0, :, :]  # need to change it for the cycle
        image = image_tensor_2D.detach().cpu().numpy()
        image[image > (image.min()+0.005)] = 1
        image[image < 1] = 0
        xx = ndimage.morphology.binary_dilation(image, iterations=2).astype(int)
        xxx = ndimage.morphology.binary_fill_holes(xx).astype(int)
        labels_mask = measure.label(xxx)
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=True)
        if len(regions) > 1:
            for rg in regions[1:]:
                labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
        labels_mask[labels_mask != 0] = 1
        mask[i] = torch.as_tensor(labels_mask, dtype=torch.float, device=torch.device('cuda:0')).unsqueeze(0)
        # mask=labels_mask
    return mask

def keep_largest_mask_black(image_tensor):
    mask = image_tensor.clone()
    #mask_black = keep_largest_mask(mask)
    for i in range(180):
        image_tensor_2D = image_tensor[i, :, :, :]  # need to change it for the cycle
        #image_tensor_2D = image_tensor[i, 0, :, :]
        image = image_tensor_2D.detach().cpu().numpy()
        image[image > (image.min()+0.005)] = 1
        image[image < 1] = 0
        mask[i] = torch.as_tensor(image, dtype=torch.float, device=torch.device('cuda:0')).unsqueeze(0)
    return mask

def keep_largest_mask_black_new(image):
    mask = image.clone()
    white = torch.ones((1, image.size()[-1], image.size()[-1]), requires_grad=False,
                       dtype=torch.float).to("cuda:0")
    black = torch.zeros((1, image.size()[-1], image.size()[-1]), requires_grad=False,
                        dtype=torch.float).to("cuda:0")
    for i in range(mask.size()[0]):
        mask[i] = torch.where(mask[i] >= (mask[i].min() + 0.1), torch.where(mask[i] < (mask[i].min() + 0.1), mask[i], white),
                           black)
    return mask

def keep_largest_mask_img(image_tensor):
    image_tensor_2D = image_tensor[0, 0, :, :]
    image = image_tensor_2D.detach().cpu().numpy()
    #prova = np_to_img(image, 'L')
    image[image > (image.min()+0.005)] = 1
    image[image < 1] = 0
    xx = ndimage.morphology.binary_dilation(image, iterations=2).astype(int)
    xxx = ndimage.morphology.binary_fill_holes(xx).astype(int)
    labels_mask = measure.label(xxx)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
    labels_mask[labels_mask != 0] = 1
    # mask = torch.as_tensor(labels_mask, dtype=torch.float, device=torch.device('cuda:0')).unsqueeze(0)
    # mask_img = tensor_to_img2(mask)
    mask_img = np_to_img(labels_mask, 'L')
    # mask=labels_mask
    return mask_img


def tensor_to_img2(tensor):
    y = tensor.detach().cpu().numpy()
    image = y[0, :, :] # ho in input da label_mask un vettore 1x128x128
    import sys
    ee = sys.float_info.epsilon
    I8 = (((image - image.min()) / (image.max() - image.min() + ee)) * 255.9).astype(np.uint8)
    img = Image.fromarray(I8, 'L')

    # image = (image - image.min()) / (image.max() - image.min())
    # image = (image * 255).astype(np.uint8)
    # img = Image.fromarray(image, 'L')
    return img

def keep_largest_mask_img_black(image_tensor):
    image_tensor_2D = image_tensor[0, :, :]
    image = image_tensor_2D.detach().cpu().numpy()
    image[image >= (image.min()+0.1)] = 1
    image[image < (image.min()+0.1)] = 0
    mask_img = np_to_img(image, 'L')
    return mask_img
