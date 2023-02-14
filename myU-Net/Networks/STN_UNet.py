import torch.nn as nn
import torch
from Networks.STN import net as STN
from Networks.UNet2D import net as UNet2D
from Networks.STN_CROP import net as STN_CROP

class STN_UNet2D(nn.Module):
    def __init__(self, ngpu, channel_dim, patch_size,mode='affine'):
        super(STN_UNet2D, self).__init__()
        self.ngpu = ngpu
        self.mode = mode
        self.stn1 = STN(ngpu,2,patch_size,mode)
        self.unet2d = UNet2D(ngpu, channel_dim)

    def forward(self, input):
        # for now it only supports one GPU
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu is 1:
            x_affine,theta = self.stn1(input)
            output = self.unet2d(x_affine)

        else:
            print('For now we only support one GPU')

        return output, x_affine, theta


class BB_UNet2D(nn.Module):
    def __init__(self, ngpu, channel_dim, patch_size):
        super(BB_UNet2D, self).__init__()
        self.ngpu = ngpu
        self.stn2 = STN_CROP(ngpu, 1,patch_size)
        self.unet2d = UNet2D(ngpu, channel_dim)

    def forward(self, input, mode='normal'):
        # for now it only supports one GPU
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu is 1:
            crop,theta_crop = self.stn2(input, mode)
            output = self.unet2d(crop)

        else:
            print('For now we only support one GPU')

        return output, crop, theta_crop