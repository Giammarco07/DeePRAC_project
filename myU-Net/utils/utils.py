import numpy as np
from scipy import ndimage
from skimage import measure
import torch as th
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from collections import OrderedDict

def keep_largest(image, mode = '2D'):
    mask = np.copy(image)
    mask[mask > (mask.min() + 0.05)] = 1
    mask[mask < 1] = 0
    xx = ndimage.morphology.binary_dilation(mask, iterations=1).astype(int)
    xxx = ndimage.morphology.binary_fill_holes(xx).astype(int)
    labels_mask = measure.label(xxx)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            if mode == '2D':
                labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
            else:
                labels_mask[rg.coords[:, 0], rg.coords[:, 1], rg.coords[:, 2]] = 0
    labels_mask[labels_mask != 0] = 1
    largest = labels_mask * image
    largest[labels_mask == 0] = image.min()
    return largest

def keep_largest_mask(image_original, minn = None, mode = '2D'):
    image = np.copy(image_original)
    if minn==None:
        minn = image.min()
    image[image > (minn + 0.005)] = 1
    image[image < 1] = 0
    xx = ndimage.morphology.binary_dilation(image, iterations=2).astype(int)  # default int=2
    xxx = ndimage.morphology.binary_fill_holes(xx).astype(int)
    labels_mask = measure.label(xxx)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            if mode == '2D':
                labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
            else:
                labels_mask[rg.coords[:, 0], rg.coords[:, 1], rg.coords[:, 2]] = 0
    labels_mask[labels_mask != 0] = 1
    mask = labels_mask
    return mask

def keep_largest_mask_torch(image):
    mask = image.clone()
    white = torch.ones((1, image.size()[-1], image.size()[-1]), requires_grad=False,
                       dtype=torch.float).to("cuda:0")
    black = torch.zeros((1, image.size()[-1], image.size()[-1]), requires_grad=False,
                        dtype=torch.float).to("cuda:0")
    for i in range(mask.size()[0]):
        mask[i] = torch.where(mask[i] >= (mask[i].min() + 0.005), torch.where(mask[i] < (mask[i].min() + 0.005), mask[i], white),
                           black)
    return mask

def inv_affine(output_affine, theta,padding='border', osize=None):
    if osize is None:
        osize=output_affine.size()
    new = torch.tensor([0, 0, 1], dtype=torch.float32,requires_grad=True).repeat(theta.size()[0], 1, 1).to("cuda")
    try:
        theta_inv = torch.inverse(torch.cat((theta, new), 1))
    except RuntimeError:
        print('non-invertible matrix somewhere in the batch -- reset to identity matrix')
        theta_inv = torch.tensor([[1, 0, 0],[0, 1, 0]], dtype=torch.float, requires_grad=True).repeat(theta.size()[0], 1, 1).to("cuda")
    grid = F.affine_grid(theta_inv[:, :2, :3], osize)
    output = F.grid_sample(output_affine, grid, mode='nearest',padding_mode=padding)
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

def referencef3(img,seg):
    theta_crop = torch.zeros((img.size()[0],3,4),requires_grad=False, dtype=torch.float).to(torch.device("cuda"))
    for i in range(img.size()[0]):
        seg_crop = torch.nonzero(seg[i], as_tuple=True)
        z = seg_crop[0]
        x = seg_crop[1]
        y = seg_crop[2]
        if x.nelement() != 0:
            theta_crop[i, 0, 0] = ((y.max() - y.min()) / (img.size()[3] * 1.0))  # z2-z1/d
            theta_crop[i, 0, 3] = ((y.max() + y.min()) / (img.size()[3] * 1.0)) - 1  # z2+z1/d - 1
            theta_crop[i, 1, 1] = ((x.max() - x.min()) / (img.size()[4] * 1.0))  # x2-x1/w
            theta_crop[i, 1, 3] = ((x.max() + x.min()) / (img.size()[4] * 1.0)) - 1  # x2+x1/w - 1
            theta_crop[i, 2, 2] = ((z.max() - z.min()) / (img.size()[2] * 1.0))  # y2-y1/h
            theta_crop[i, 2, 3] = ((z.max() + z.min()) / (img.size()[2] * 1.0)) - 1  # x2+x1/w - 1

        else:
            theta_crop[i, 0, 0] = 1.0  # x2-x1/w
            theta_crop[i, 0, 3] = 0.0  # x2+x1/w - 1
            theta_crop[i, 1, 1] = 1.0  # y2-y1/h
            theta_crop[i, 1, 3] = 0.0  # y2+y1/h -1
            theta_crop[i, 2, 2] = 1.0  # z2-z1/h
            theta_crop[i, 2, 3] = 0.0  # z2+z1/h -1

    grid_cropped = F.affine_grid(theta_crop, img.size(), align_corners=False)
    img_crop = F.grid_sample(img, grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"
    img_crop.requires_grad=False

    return img_crop, theta_crop
    
    
def referencef3D(img,seg):
    theta_crop = torch.zeros((img.size()[0],3,4),requires_grad=False, dtype=torch.float).to(torch.device("cuda"))
    for jj in range(img.size()[0]):
        seg_crop = torch.nonzero(seg[jj], as_tuple=True)
        y = seg_crop[0]  # depth: axis z
        x = seg_crop[1]  # row: axis y
        z = seg_crop[2]  # col: axis x
        if x.nelement() == 0 or x.min() == x.max() or y.min() == y.max() or z.min() == z.max():
            theta_crop[jj, 0, 0] = 1
            theta_crop[jj, 0, 3] = 0
            theta_crop[jj, 1, 1] = 1
            theta_crop[jj, 1, 3] = 0
            theta_crop[jj, 2, 2] = 1
            theta_crop[jj, 2, 3] = 0
        else:
            theta_crop[jj, 0, 0] = ((z.max() - z.min()) / (img.size()[3] * 1.0))  # x2-x1/w
            theta_crop[jj, 0, 3] = ((z.max() + z.min()) / (img.size()[3] * 1.0)) - 1  # x2+x1/w - 1
            theta_crop[jj, 1, 1] = ((x.max() - x.min()) / (img.size()[4] * 1.0))  # y2-y1/h
            theta_crop[jj, 1, 3] = ((x.max() + x.min()) / (img.size()[4] * 1.0)) - 1  # x2+x1/w - 1
            theta_crop[jj, 2, 2] = ((y.max() - y.min()) / (img.size()[2] * 1.0))  # z2-z1/h
            theta_crop[jj, 2, 3] = ((y.max() + y.min()) / (img.size()[2] * 1.0)) - 1  # z2+z1/w - 1

    grid_cropped = F.affine_grid(theta_crop, img.size(), align_corners=False)
    img_crop = F.grid_sample(img, grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"
    img_crop.requires_grad=False

    return img_crop, theta_crop


def isSafe(x, y, z, processed, seg,tgt):
    return (x >= 0) and (x < processed.shape[0]) and \
           (y >= 0) and (y < processed.shape[1]) and \
           (z >= 0) and (z < processed.shape[2]) and \
           (seg[x,y,z] == 0)

from collections import deque

def get26n(x, y, z, processed, seg,tgt):
    # Below lists detail all 26 possible movements from a cell
    row = [-1, -1, -1, 0, 1, 0, 1, 1, -1, -1, -1, 0, 1, 0, 1, 1, 0, -1, -1, -1, 0, 1, 0, 1, 1, 0]
    col = [-1, 1, 0, -1, -1, 1, 0, 1, -1, 1, 0, -1, -1, 1, 0, 1, 0, -1, 1, 0, -1, -1, 1, 0, 1, 0]
    slc = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    q = deque()
    for p in range(26):
        for i in range(1,3):
            for j in range(1, 3):
                for k in range(1, 3):
                    # skip if the location is invalid  or it is already in seg
                    if isSafe(x + row[p]*i, y + col[p]*j, z + slc[p]*k, processed, seg,tgt):
                        q.append((x + row[p]*i, y + col[p]*j, z + slc[p]*k))

    q = list(OrderedDict((tuple(x), x) for x in q).values())
    return q


def region_growing(img, seg, tgt, seedss, sdes):
    for i in range(1,seg.max().astype(int)+1):
        print('structure:', i)
        sde = sdes[i-1]
        print(sde)
        seeds = seedss[i-1]
        print(len(seeds))
        processed = []
        newseeds = []
        points = 0
        while (len(seeds) > 0):
            pix = seeds[0]
            value = img[pix[0], pix[1], pix[2]]

            for coord in get26n(pix[0], pix[1], pix[2], img,seg, tgt):
                if img[coord[0], coord[1], coord[2]]<(value+sde) and img[coord[0], coord[1], coord[2]]>(value-sde):
                    seg[coord[0], coord[1], coord[2]] = i
                    if not coord in processed:
                        newseeds.append(coord)
                    processed.append(coord)
            seeds.pop(0)
            while (len(newseeds)>0) and points<5000:
                pix = newseeds[0]
                points += 1
                for coord in get26n(pix[0], pix[1], pix[2], img, seg, tgt):
                    if img[coord[0], coord[1], coord[2]] < (value + sde) and img[coord[0], coord[1], coord[2]] > (
                            value - sde):
                        seg[coord[0], coord[1], coord[2]] = i
                        if not coord in processed:
                            newseeds.append(coord)
                        processed.append(coord)
                newseeds.pop(0)

            newseeds.clear()
            points = 0
    return seg
