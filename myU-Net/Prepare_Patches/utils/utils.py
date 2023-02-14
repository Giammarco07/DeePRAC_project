from scipy import ndimage
from skimage import measure
import torch as th
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

def keep_largest(image):
    image[image > image.min()] = 1
    image[image < 1] = 0
    xx = ndimage.morphology.binary_dilation(image, iterations=2).astype(int)
    xxx = ndimage.morphology.binary_fill_holes(xx).astype(int)
    labels_mask = measure.label(xxx)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1
    mask = labels_mask
    return mask

def inv_affine(output_affine, theta):
    new = torch.tensor([0, 0, 1], dtype=torch.float32,requires_grad=True).repeat(theta.size()[0], 1, 1).to("cuda")
    try:
        theta_inv = torch.inverse(torch.cat((theta, new), 1))
    except RuntimeError:
        print('non-invertible matrix somewhere in the batch -- reset to identity matrix')
        theta_inv = torch.tensor([[1, 0, 0],[0, 1, 0]], dtype=torch.float, requires_grad=True).repeat(theta.size()[0], 1, 1).to("cuda")
    grid = F.affine_grid(theta_inv[:, :2, :3], output_affine.size())
    output = F.grid_sample(output_affine, grid, mode='nearest')
    return output

def inv_affine3d(output_affine, theta):
    new = torch.tensor([0, 0, 0, 1], dtype=torch.bfloat16,requires_grad=True).repeat(theta.size()[0], 1, 1).to("cuda")
    try:
        theta_inv = torch.inverse(torch.cat((theta, new), 1))
    except RuntimeError:
        print('non-invertible matrix somewhere in the batch -- reset to identity matrix')
        theta_inv = torch.tensor([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float, requires_grad=True).repeat(theta.size()[0], 1, 1).to("cuda")

    grid = F.affine_grid(theta_inv[:, :3, :4], output_affine.size())
    output = F.grid_sample(output_affine, grid.type(torch.float), mode='nearest')
    return output

def referencef(img,seg):
    theta_crop = torch.zeros((img.size()[0],2,3),requires_grad=False, dtype=torch.float).to(torch.device("cuda"))
    for i in range(img.size()[0]):
        seg_crop = torch.nonzero(seg[i], as_tuple=True)
        y = seg_crop[0]
        x = seg_crop[1]
        if x.nelement() == 0 or x.min() == x.max():
            theta_crop[i, 0, 0] = 1.0  # x2-x1/w
            theta_crop[i, 0, 2] = 0.0  # x2+x1/w - 1
            theta_crop[i, 1, 1] = 1.0  # y2-y1/h
            theta_crop[i, 1, 2] = 0.0  # y2+y1/h -1
        elif y.nelement() == 0 or y.min() == y.max():
            theta_crop[i, 0, 0] = 1.0  # x2-x1/w
            theta_crop[i, 0, 2] = 0.0  # x2+x1/w - 1
            theta_crop[i, 1, 1] = 1.0  # y2-y1/h
            theta_crop[i, 1, 2] = 0.0  # y2+y1/h -1
        elif ((x.max() - x.min()) < (img.size()[2] / 4)) and ((y.max() - y.min()) >= (img.size()[3] / 4)):
            add = (y.max() - y.min()) - (x.max() - x.min())
            theta_crop[i, 0, 0] = ((x.max() - x.min() + add) / (img.size()[2] * 1.0))  # x2-x1/w
            if x.min() < (add / 2):
                theta_crop[i, 0, 2] = ((x.max() + add - x.min()) / (img.size()[2] * 1.0)) - 1  # x2+x1/w - 1
            elif x.max() > (img.size()[2] - (add / 2)):
                theta_crop[i, 0, 2] = ((img.size()[2] + x.min() - add + img.size()[2] - x.max()) / (
                        img.size()[2] * 1.0)) - 1  # x2+x1/w - 1
            else:
                theta_crop[i, 0, 2] = ((x.max() + x.min()) / (img.size()[2] * 1.0)) - 1  # x2+x1/w - 1
            theta_crop[i, 1, 1] = ((y.max() - y.min()) / (img.size()[3] * 1.0))  # y2-y1/h
            theta_crop[i, 1, 2] = ((y.max() + y.min()) / (img.size()[3] * 1.0)) - 1  # y2+y1/h -1
        elif ((x.max() - x.min()) >= (img.size()[2] / 4)) and ((y.max() - y.min()) < (img.size()[3] / 4)):
            add = (x.max() - x.min()) - (y.max() - y.min())
            theta_crop[i, 0, 0] = ((x.max() - x.min()) / (img.size()[2] * 1.0))  # x2-x1/w
            theta_crop[i, 0, 2] = ((x.max() + x.min()) / (img.size()[2] * 1.0)) - 1  # x2+x1/w - 1
            theta_crop[i, 1, 1] = ((y.max() - y.min() + add) / (img.size()[3] * 1.0))  # y2-y1/h
            if y.min() < (add / 2):
                theta_crop[i, 1, 2] = ((y.max() + add - y.min()) / (img.size()[3] * 1.0)) - 1  # x2+x1/w - 1
            elif y.max() > (img.size()[3] - (add / 2)):
                theta_crop[i, 1, 2] = ((img.size()[3] + y.min() - add + img.size()[3] - y.max()) / (
                        img.size()[3] * 1.0)) - 1  # x2+x1/w - 1
            else:
                theta_crop[i, 1, 2] = ((y.max() + y.min()) / (img.size()[3] * 1.0)) - 1  # x2+x1/w - 1
        elif ((x.max() - x.min()) < (img.size()[2] / 4)) and ((y.max() - y.min()) < (img.size()[3] / 4)):
            addx = (img.size()[2] / 4) - (x.max() - x.min())
            addy = (img.size()[3] / 4) - (y.max() - y.min())
            theta_crop[i, 0, 0] = ((x.max() - x.min() + addx) / (img.size()[2] * 1.0))  # x2-x1/w
            if x.min() < (addx / 2):
                theta_crop[i, 0, 2] = ((x.max() + addx - x.min()) / (img.size()[2] * 1.0)) - 1  # x2+x1/w - 1
            elif x.max() > (img.size()[2] - (addx / 2)):
                theta_crop[i, 0, 2] = ((img.size()[2] + x.min() - addx + img.size()[2] - x.max()) / (
                        img.size()[2] * 1.0)) - 1  # x2+x1/w - 1
            else:
                theta_crop[i, 0, 2] = ((x.max() + x.min()) / (img.size()[2] * 1.0)) - 1  # x2+x1/w - 1
            theta_crop[i, 1, 1] = ((y.max() - y.min() + addy) / (img.size()[3] * 1.0))  # y2-y1/h
            if y.min() < (addy / 2):
                theta_crop[i, 1, 2] = ((y.max() + addy - y.min()) / (img.size()[3] * 1.0)) - 1  # x2+x1/w - 1
            elif y.max() > (img.size()[3] - (addy / 2)):
                theta_crop[i, 1, 2] = ((img.size()[3] + y.min() - addy + img.size()[3] - y.max()) / (
                        img.size()[3] * 1.0)) - 1  # x2+x1/w - 1
            else:
                theta_crop[i, 1, 2] = ((y.max() + y.min()) / (img.size()[3] * 1.0)) - 1  # x2+x1/w - 1
        else:
            if (y.max() - y.min()) > (x.max() - x.min()):
                add = (y.max() - y.min()) - (x.max() - x.min())
                theta_crop[i, 0, 0] = (((x.max() - x.min()) + add) / (img.size()[2] * 1.0))  # x2-x1/w
                if x.min() < (add / 2) and x.max() < (img.size()[2] - (add / 2)):
                    theta_crop[i, 0, 2] = ((x.max() + add - x.min()) / (img.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                elif x.min() > (add / 2) and x.max() > (img.size()[2] - (add / 2)):
                    theta_crop[i, 0, 2] = ((img.size()[2] + x.min() - add + img.size()[2] - x.max()) / (
                            img.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                else:
                    theta_crop[i, 0, 2] = ((x.max() + x.min()) / (img.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                theta_crop[i, 1, 1] = ((y.max() - y.min()) / (img.size()[3] * 1.0))  # y2-y1/h
                theta_crop[i, 1, 2] = ((y.max() + y.min()) / (img.size()[3] * 1.0)) - 1  # y2+y1/h -1
            else:
                add = (x.max() - x.min()) - (y.max() - y.min())
                theta_crop[i, 0, 0] = ((x.max() - x.min()) / (img.size()[2] * 1.0))  # x2-x1/w
                theta_crop[i, 0, 2] = ((x.max() + x.min()) / (img.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                theta_crop[i, 1, 1] = (((y.max() - y.min()) + add) / (img.size()[3] * 1.0))  # y2-y1/h
                if y.min() < (add / 2) and y.max() < (img.size()[3] - (add / 2)):
                    theta_crop[i, 1, 2] = ((y.max() + add - y.min()) / (img.size()[3] * 1.0)) - 1  # x2+x1/w - 1
                elif y.min() > (add / 2) and y.max() > (img.size()[3] - (add / 2)):
                    theta_crop[i, 1, 2] = ((img.size()[3] + y.min() - add + img.size()[3] - y.max()) / (
                            img.size()[3] * 1.0)) - 1  # x2+x1/w - 1
                else:
                    theta_crop[i, 1, 2] = ((y.max() + y.min()) / (img.size()[3] * 1.0)) - 1  # x2+x1/w - 1

    grid_cropped = F.affine_grid(theta_crop, img.size(), align_corners=False)
    img_crop = F.grid_sample(img, grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"
    img_crop.requires_grad=False

    return img_crop, theta_crop


def referencef2(img,seg):
    theta_crop = torch.zeros((img.size()[0],2,3),requires_grad=False, dtype=torch.float).to(torch.device("cuda"))
    img_crop = img.clone()
    for i in range(img.size()[0]):
        seg_crop = torch.nonzero(seg[i], as_tuple=True)
        x = seg_crop[0]
        y = seg_crop[1]
        if x.nelement() != 0:
            img_crop[i,0,0:x.min(),:] = img.min()
            img_crop[i, 0, (x.max()+1):img.size()[2], :] = img.min()
            img_crop[i, 0,:, 0:y.min()] = img.min()
            img_crop[i, 0,:, (y.max()+1):img.size()[3]] = img.min()
            theta_crop[i, 0, 0] = ((y.max() - y.min()) / (img.size()[3] * 1.0))  # x2-x1/w
            theta_crop[i, 0, 2] = ((y.max() + y.min()) / (img.size()[3] * 1.0)) - 1  # x2+x1/w - 1
            theta_crop[i, 1, 1] = ((x.max() - x.min()) / (img.size()[2] * 1.0))  # y2-y1/h
            theta_crop[i, 1, 2] = ((x.max() + x.min()) / (img.size()[2] * 1.0)) - 1  # x2+x1/w - 1

        else:
            theta_crop[i, 0, 0] = 1.0  # x2-x1/w
            theta_crop[i, 0, 2] = 0.0  # x2+x1/w - 1
            theta_crop[i, 1, 1] = 1.0  # y2-y1/h
            theta_crop[i, 1, 2] = 0.0  # y2+y1/h -1

    return img_crop, theta_crop