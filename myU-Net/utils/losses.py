from __future__ import division

import numpy as np
import torch
import sys
ee = sys.float_info.epsilon
smooth = 1e-5

def dice(input, target):

    num = input * target
    num = np.sum(num, axis=3)
    num = np.sum(num,axis=2)
    num = np.sum(num, axis=1)

    den1 = input
    den1 = np.sum(den1, axis=3)
    den1 = np.sum(den1, axis=2)
    den1 = np.sum(den1, axis=1)

    den2 = target
    den2 = np.sum(den2, axis=3)
    den2 = np.sum(den2,axis=2)
    den2 = np.sum(den2, axis=1)

    dice_ = ((2 * num) + ee) / (den1 + den2 + ee)
    print(dice_*100)

    dice_total = np.mean(dice_) # divide by batchsize
    dice_std = np.std(dice_)

    return dice_total, dice_std


def dice_post(input, target):
    D = np.minimum(input.shape[0], target.shape[0])
    H = np.minimum(input.shape[1], target.shape[1])
    W = np.minimum(input.shape[2], target.shape[2])
    num = input[:D,:H,:W] * target[:D,:H,:W]
    num = np.sum(num)

    den1 = input[:D,:H,:W].sum()

    den2 = target[:D,:H,:W].sum()

    dice_ = ((2 * num) + ee) / (den1 + den2 + ee)

    return dice_


def soft_dice_loss(input, target):
    num = input*target
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=2)
    if len(input.size())==5:
        num = torch.sum(num, dim=2)

    den1 = input*input
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)
    if len(input.size())==5:
        den1 = torch.sum(den1, dim=2)

    den2 = target*target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)
    if len(input.size())==5:
        den2 = torch.sum(den2, dim=2)

    dice = ((2 * num) + smooth) / (den1 + den2 + smooth + ee)
    dice = dice[:, 1:]
    dice = dice.mean()

    return dice

def soft_dice_loss_old(input, target):
    num = input*target
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=2)
    if len(input.size())==5:
        num = torch.sum(num, dim=2)

    den1 = input*input
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)
    if len(input.size())==5:
        den1 = torch.sum(den1, dim=2)

    den2 = target*target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)
    if len(input.size())==5:
        den2 = torch.sum(den2, dim=2)

    dice_ = ((2 * num) + smooth) / (den1 + den2 + smooth + ee)
    dice = dice_[:, 1:]
    dice_total = 1. - 1. * dice.mean() # divide by batchsize and channels

    return dice_total

def soft_dice_loss_old_new(input, target, gt_dis):
    num = (input*target)/(gt_dis+1)
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=2)
    if len(input.size())==5:
        num = torch.sum(num, dim=2)

    den1 = (input*input)/(gt_dis+1)
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)
    if len(input.size())==5:
        den1 = torch.sum(den1, dim=2)

    den2 = (target*target)/(gt_dis+1)
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)
    if len(input.size())==5:
        den2 = torch.sum(den2, dim=2)

    dice_ = ((2 * num) + smooth) / (den1 + den2 + smooth + ee)
    dice = dice_[:, 1:]
    dice_total = 1. - 1. * dice.mean() # divide by batchsize and channels

    return dice_total

def soft_dice_loss_batch(input, target, channel=False):
    num = input*target
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=2)
    if len(input.size())==5:
        num = torch.sum(num, dim=2)

    den1 = input*input
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)
    if len(input.size())==5:
        den1 = torch.sum(den1, dim=2)

    den2 = target*target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)
    if len(input.size())==5:
        den2 = torch.sum(den2, dim=2)

    dice_ = ((2 * torch.sum(num, dim=0)) + smooth) / (torch.sum(den1, dim=0) + torch.sum(den2, dim=0) + smooth + ee)
    if not channel:
        dice = dice_[1:]
    else:
        dice = dice_
    dice_total = 1. - 1. * dice.mean() # divide by channels

    return dice_total

def general_dice_loss_batch(input, target, channel=False):
    num1 = input*target
    num1 = torch.sum(num1, dim=2)
    num1 = torch.sum(num1, dim=2)
    if len(input.size())==5:
        num1 = torch.sum(num1, dim=2)
    num1 = torch.sum(num1, dim=0)
    
    num2 = (1-input)*(1-target)
    num2 = torch.sum(num2, dim=2)
    num2 = torch.sum(num2, dim=2)
    if len(input.size())==5:
        num2 = torch.sum(num2, dim=2)
    num2 = torch.sum(num2, dim=0)
    
    den1 = input+target
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)
    if len(input.size())==5:
        den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=0)
    
    den2 = 2-input-target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)
    if len(input.size())==5:
        den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=0)
    
    wb = torch.sum(1-target,dim=2)   
    wb = torch.sum(wb,dim=2)   
    if len(input.size())==5:
        wb = torch.sum(wb, dim=2)
    wb = 1/((wb**2)+ee)
    wb = torch.sum(wb,dim=0)   
    
    wg = torch.sum(target,dim=2)   
    wg = torch.sum(wg,dim=2)   
    if len(input.size())==5:
        wg = torch.sum(wg, dim=2)
    wg = 1/((wg**2)+ee)
    wg = torch.sum(wg,dim=0)  
     
    dice_ = (2 * (wg*num1+wb*num2))/(wg*den1 + wb*den2 + ee)
    if not channel:
        dice = dice_[1:]
    else:
        dice = dice_
    dice_total = 1. - 1. * dice.mean() # divide by channels

    return dice_total

def fnr(input, target, channel=False):
    num = (1-input)*target
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=2)
    if len(input.size())==5:
        num = torch.sum(num, dim=2)

    den = target
    den = torch.sum(den, dim=2)
    den = torch.sum(den, dim=2)
    if len(input.size())==5:
        den = torch.sum(den, dim=2)

    fn_ = (torch.sum(num, dim=0) + smooth) / (torch.sum(den, dim=0) + smooth + ee)
    if not channel:
        fn = fn_[1:]
    else:
        fn = fn_
    fn_total = 1. * fn.mean() # divide by batchsize and channels

    return fn_total

def dice_loss(input, target):
    num = input * target
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=2)
    if len(input.size())==5:
        num = torch.sum(num, dim=2)

    den1 = input
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)
    if len(input.size())==5:
        den1 = torch.sum(den1, dim=2)

    den2 = target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)
    if len(input.size())==5:
        den2 = torch.sum(den2, dim=2)

    dice_ = ((2 * num) + ee) / (den1 + den2 + ee)
    dice = dice_[:, 1:]
    dice_total = dice.mean()  # divide by batchsize and channels

    return dice_total

def dice_loss_val(input, target):
    num = input * target
    num = torch.sum(num, dim=1)
    num = torch.sum(num, dim=1)
    if len(input.size()) == 4:
        num = torch.sum(num, dim=1)

    den1 = input
    den1 = torch.sum(den1, dim=1)
    den1 = torch.sum(den1, dim=1)
    if len(input.size()) == 4:
        den1 = torch.sum(den1, dim=1)

    den2 = target
    den2 = torch.sum(den2, dim=1)
    den2 = torch.sum(den2, dim=1)
    if len(input.size()) == 4:
        den2 = torch.sum(den2, dim=1)

    dice = ((2 * num) + ee) / (den1 + den2 + ee)

    dice_total = dice.mean()  # divide by batchsize

    return dice_total

def dice_loss_val_new(input, target):
    num = input * target
    num = torch.sum(num, dim=1)
    num = torch.sum(num, dim=1)
    if len(input.size()) == 4:
        num = torch.sum(num, dim=1)

    den1 = input
    den1 = torch.sum(den1, dim=1)
    den1 = torch.sum(den1, dim=1)
    if len(input.size()) == 4:
        den1 = torch.sum(den1, dim=1)

    den2 = target
    den2 = torch.sum(den2, dim=1)
    den2 = torch.sum(den2, dim=1)
    if len(input.size()) == 4:
        den2 = torch.sum(den2, dim=1)

    dice = ((2 * num) + ee) / (den1 + den2 + ee)

    n = ee
    dice_total = torch.zeros(1).to(device='cuda')
    for x in range(dice.size(0)):
        if dice[x] != 1:
            dice_total += dice[x]
            n += 1

    dice_total /= n

    return dice_total

def compute_dtm(img_gt, out_shape):
    from scipy.ndimage import distance_transform_edt as distance
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, c, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = np.rint(posdis)

    return fg_dtm

def compute_ddt(img_gt, out_shape, k):
    from scipy.ndimage import distance_transform_edt as distance
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, 1, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(img_gt.shape[0]): # batch size
        for c in range(img_gt.shape[1]):
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                posdis = np.rint(distance(posmask))
                mask = posdis>0
                fg_dtm[b] += (c*k*mask + posdis)

    return fg_dtm

bce = torch.nn.BCELoss().to(device='cuda')

sig_bce = torch.nn.BCEWithLogitsLoss().to(device='cuda')

ce = torch.nn.NLLLoss().to(device='cuda')

L2 = torch.nn.MSELoss().to(device='cuda')

L1 = torch.nn.L1Loss().to(device='cuda')

L1smooth = torch.nn.SmoothL1Loss().to(device='cuda')

def c_loss(pred,target,seg_vessels):
    loss = torch.where(torch.abs(target-pred)<1,0.5*((target-pred)**2),torch.abs(target-pred)-0.5)
    loss = loss / (target**2)

    return torch.mean(loss[seg_vessels==1])

def top_loss(pred,seg,center_vessels, r=15, K=3,alpha=1/15,gamma=1/3):
    grid = torch.stack(torch.meshgrid(torch.arange(pred.size()[-3]), torch.arange(pred.size()[-2]), torch.arange(pred.size()[-1])), dim=-1)
    dist = torch.zeros((pred.size()[-3],pred.size()[-2],pred.size()[-1]))
    l = 0.
    n = 0
    for pos in torch.nonzero(center_vessels.squeeze(1)):
        for g in grid:
            dist[g] = (pos[0] - g).pow(2).sum().sqrt()
        dist[dist>r] = 0
        for d in torch.nonzero(dist):
            d_ = [pos[0],d[0],d[1],d[2]]
            if d_ == pos:
                continue
            else:
                if seg[d_]==seg[pos]:
                    l += L1smooth(L2(pred[pos[0],:,pos[2],pos[3],pos[4]],pred[pos[0],:,d[0],d[1],d[2]]),alpha*dist[d])
                    n += 1
                else:
                    l += gamma*(K-L2(pred[pos[0],:,pos[2],pos[3],pos[4]],pred[pos[0],:,d[0],d[1],d[2]]))
                    n += 1

    return l/n

