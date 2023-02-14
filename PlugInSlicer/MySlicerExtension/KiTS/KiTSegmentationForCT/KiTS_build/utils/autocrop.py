import numpy as np
import torch
from utils.patches import adjust_dim
from utils.pre_processing import rescaled
from skimage.transform import resize
import torch.nn.functional as F
import sys
ee = sys.float_info.epsilon
from utils.cropping import crop_to_nonzero
def security(im_tensor, sz_crop):
    for i in range(im_tensor.size()[0]):
        # adding safety margins of 0.2 of the size
        addz = int(im_tensor.size()[2] * 0.2)
        addx = int(im_tensor.size()[3] * 0.2)
        addy = int(im_tensor.size()[4] * 0.2)
        if (sz_crop[i, 1] - sz_crop[i, 0]) < (im_tensor.size()[2] - addz):
            if sz_crop[i, 0] < (addz / 2):
                sz_crop[i, 1] += (addz / 2)
            elif sz_crop[i, 1] > (im_tensor.size()[2] - (addz / 2)):
                sz_crop[i, 0] -= (addz / 2)
            else:
                sz_crop[i, 1] += (addz / 2)
                sz_crop[i, 0] -= (addz / 2)
        if (sz_crop[i, 3] - sz_crop[i, 2]) < (im_tensor.size()[3] - addx):
            if sz_crop[i, 2] < (addx / 2):
                sz_crop[i, 3] += (addx / 2)
            elif sz_crop[i, 3] > (im_tensor.size()[3] - (addx / 2)):
                sz_crop[i, 2] -= (addx / 2)
            else:
                sz_crop[i, 3] += (addx / 2)
                sz_crop[i, 2] -= (addx / 2)
        if (sz_crop[i, 5] - sz_crop[i, 4]) < (im_tensor.size()[4] - addy):
            if sz_crop[i, 4] < (addy / 2):
                sz_crop[i, 5] += (addy / 2)
            elif sz_crop[i, 5] > (im_tensor.size()[4] - (addy / 2)):
                sz_crop[i, 4] -= (addy / 2)
            else:
                sz_crop[i, 5] += (addy / 2)
                sz_crop[i, 4] -= (addy / 2)

    return sz_crop

def theta(im_tensor, sz_crop):
    theta_crop = torch.zeros((im_tensor.size()[0], 3, 4), requires_grad=True, dtype=torch.float).to(
        torch.device("cuda"))
    theta_crop[:, 0, 0] = ((sz_crop[:, 1] - sz_crop[:, 0]) / (im_tensor.size()[2] * 1.0)) + 1e-20  # x2-x1/w
    theta_crop[:, 0, 3] = ((sz_crop[:, 1] + sz_crop[:, 0]) / (im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
    theta_crop[:, 1, 1] = ((sz_crop[:, 3] - sz_crop[:, 2]) / (im_tensor.size()[3] * 1.0)) + 1e-20  # y2-y1/h
    theta_crop[:, 1, 3] = ((sz_crop[:, 3] + sz_crop[:, 2]) / (im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h -1
    theta_crop[:, 2, 2] = ((sz_crop[:, 5] - sz_crop[:, 4]) / (im_tensor.size()[4] * 1.0)) + 1e-20  # z2-z1/d
    theta_crop[:, 2, 3] = ((sz_crop[:, 5] + sz_crop[:, 4]) / (im_tensor.size()[4] * 1.0)) - 1  # z2+z1/d -1

    return theta_crop

def theta_new(sz_crop):
    theta_crop = torch.zeros((3, 2), requires_grad=True, dtype=torch.float).to(
        torch.device("cuda"))
    theta_crop[1, 0] = (sz_crop[:, 1] - sz_crop[:, 0]) / 2
    theta_crop[1, 1] = (sz_crop[:, 1] + sz_crop[:, 0]) / 2
    theta_crop[0, 0] = (sz_crop[:, 3] - sz_crop[:, 2]) / 2
    theta_crop[0, 1] = (sz_crop[:, 3] + sz_crop[:, 2]) / 2
    theta_crop[2, 0] = (sz_crop[:, 5] - sz_crop[:, 4]) / 2
    theta_crop[2, 1] = (sz_crop[:, 5] + sz_crop[:, 4]) / 2

    return theta_crop

def stncrop(img,patch_size,preprocessing,net,device):
    if img.shape[2] > img.shape[0]:
        d = img.shape[2] - img.shape[0]
        print(d)
        img2 = adjust_dim(img, img.shape[0] + d, img.shape[1] + d, img.shape[2])
        a=1
    elif img.shape[2] < img.shape[0]:
        d = img.shape[0] - img.shape[2]
        print(d)
        img2 = adjust_dim(img, img.shape[0], img.shape[1], img.shape[2] + d)
        a=2
    else:
        img2 = img
        a = 0

    image = resize(img2, patch_size, order=1, mode='constant', cval=img2.min(), anti_aliasing=True)
    image = rescaled(image,preprocessing)
    image = np.expand_dims(np.expand_dims(image, 0),0)
    t_image = torch.from_numpy(image).type('torch.FloatTensor').to(device)
    print(t_image.size())

    net.eval()
    with torch.no_grad():
        sz_c = net(t_image)
        print(sz_c)
        sz_cs = security(t_image,sz_c)
        print(sz_cs)
        sz_crop = ((sz_cs/patch_size[0])*img2.shape[-1]).type(torch.int64)
        print(sz_crop)

        if a==1:
            sz_crop[:,0] = max(0,sz_crop[0,0]-d//2)
            sz_crop[:,2] = max(0,sz_crop[0,2]-d//2)
            sz_crop[:,1] = min(img.shape[0],sz_crop[0,1]-d//2)
            sz_crop[:,3] = min(img.shape[1],sz_crop[0,3]-d//2)

        elif a==2:
            sz_crop[:,4] = max(0,sz_crop[0,4]-d//2)
            sz_crop[:,5] = min(img.shape[2],sz_crop[0,5]-d//2)

        else:
            pass
        print(sz_crop)
        theta_crop = theta_new(sz_crop)



    return theta_crop.data.cpu().numpy()






