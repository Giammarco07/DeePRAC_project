from pathlib import Path
import os
from skimage.morphology import skeletonize, skeletonize_3d
import nibabel as nib
from PIL import Image
import numpy as np
import sys
ee = sys.float_info.epsilon
import matplotlib.pyplot as plt
from islands3d import islands3d, direction, plane, minimo,forcing
from utils.pre_processing import resample_nii, rescaled, load_nii, save_nii
import cc3d
import sknw

def end_bif_points(skeleton3d):
    end = np.copy(skeleton3d)
    bif = np.copy(skeleton3d)
    for i in range(1,skeleton3d.shape[0]-1):
        for j in range(1,skeleton3d.shape[1]-1):
            for k in range(1,skeleton3d.shape[2]-1):
                if skeleton3d[i-1:i+2,j-1:j+2,k-1:k+2].sum()!=2:
                    end[i,j,k]=0
                if skeleton3d[i-1:i+2,j-1:j+2,k-1:k+2].sum()<=3:
                    bif[i,j,k]=0

    return end,bif

def bif_conv(skeleton):
    import torch
    import torch.nn.functional as F
    bif = np.copy(skeleton)
    skel = torch.as_tensor(skeleton, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    weights = torch.tensor([[1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.]])
    weights = weights.view(1, 1, 1, 3, 3).repeat(1, 1, 3, 1, 1)
    output = (F.conv3d(skel, weights, stride = 1, padding = 1)).squeeze(0).squeeze(0).data.numpy()
    bif[output<=3] = 0

    return bif



def combinator(*args):
    args = list(args)
    n = args.pop()
    cur = np.arange(n)
    cur = cur[:, None]

    while args:
        d = args.pop()
        cur = np.kron(np.ones((d, 1)), cur)
        front = np.arange(d).repeat(n)[:, None]
        cur = np.column_stack((front, cur))
        n *= d

    return cur


path = '/tsi/clusterhome/glabarbera/unet3d'
task = 'Task208_NECKER'
dirpath = Path(path + '/nnUNet_preprocessed/' + task + '/nnUNetData_plans_v2.1_stage1')
folder = path + '/nnUNet_preprocessed/' + task + '/nnUNetData_plans_v2.1_stage1/Patches_perp_3D2_new'

if not os.path.exists(folder):
    os.mkdir(folder)

import time
start_time = time.time()
for name in dirpath.iterdir():
    if (name.is_file()) & (name.suffix == '.npz'):
        print(str(name)[-19:-4])
        im = np.load(name)
        img = im[im.files[0]][0, :, :, :].astype(np.float32) 
        seg_r = im[im.files[0]][1, :, :, :].astype(np.float32) 


        print(seg_r.shape)
        print(np.unique(seg_r))
        n_struct = seg_r.max().astype(int)
        print(img.shape)
        print(img.max(),img.min())

        # 0. BIFURCATION POINTS AND ISLANDS
        for struct in range(1,n_struct+1):
            print('structure:', struct)
            dat = seg_r == struct
            skel = skeletonize(dat)
            skel = skel / skel.max()


            graph = sknw.build_sknw(skel)
            for (s, e) in graph.edges():

                print('from',s,'to',e)
                ps = graph[s][e]['pts']
                nps = ps.shape[0]
                print('points:',nps)
                total = int(nps//16 + 2)
                print('patches:', total)
                start = ps[0]
                end = ps[-1]
                patches = np.full((total, 2, 32, 64, 64), img.min(), dtype=np.float32)
                patches[:, 1, :, :, :] = 0

                patch =  img[start[0]-16:start[0]+16,start[1]-32:start[1]+32,start[2]-32:start[2]+32]
                segpatch = seg_r[start[0] - 16:start[0] + 16, start[1] - 32:start[1] + 32, start[2] - 32:start[2] + 32]
                x,y,z = 16-patch.shape[0]//2, 32-patch.shape[1]//2, 32-patch.shape[2]//2
                patches[0,0, x:x+patch.shape[0],y:y+patch.shape[1],z:z+patch.shape[2]] = patch
                patches[0,1,x:x+patch.shape[0],y:y+patch.shape[1],z:z+patch.shape[2]] = segpatch

                if nps>15:
                    ind = 15
                    idx = 1
                    while ind<nps:
                        cord = ps[ind]
                        patch = img[cord[0] - 16:cord[0] + 16, cord[1] - 32:cord[1] + 32, cord[2] - 32:cord[2] + 32]
                        segpatch = seg_r[cord[0] - 16:cord[0] + 16, cord[1] - 32:cord[1] + 32, cord[2] - 32:cord[2] + 32]
                        x, y, z = 16 - patch.shape[0] // 2, 32 - patch.shape[1] // 2, 32 - patch.shape[2] // 2
                        patches[idx, 0, x:x+patch.shape[0],y:y+patch.shape[1],z:z+patch.shape[2]] = patch
                        patches[idx, 1, x:x+patch.shape[0],y:y+patch.shape[1],z:z+patch.shape[2]]  = segpatch

                        ind += 16
                        idx += 1

                patch = img[end[0]-16:end[0]+16,end[1]-32:end[1]+32,end[2]-32:end[2]+32]
                segpatch = seg_r[end[0] - 16:end[0] + 16, end[1] - 32:end[1] + 32, end[2] - 32:end[2] + 32]
                x,y,z = 16-patch.shape[0]//2, 32-patch.shape[1]//2, 32-patch.shape[2]//2
                patches[-1,0,x:x+patch.shape[0],y:y+patch.shape[1],z:z+patch.shape[2]] = patch
                patches[-1,1,x:x+patch.shape[0],y:y+patch.shape[1],z:z+patch.shape[2]] = segpatch

                for z in range(total):
                    if np.count_nonzero(patches[z, 1, :, :, :])==0:
                    	continue
                    np.savez(folder + '/' + str(name)[-19:-4] + '_' + str(struct) + '_' + str(s) + str(e) + '_' + str(z) + '.npz',
                         patches[z, :, :, :, :])

                del patches




