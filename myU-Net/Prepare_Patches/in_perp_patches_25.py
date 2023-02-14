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

# path = '/home/infres/glabarbera/nnunet'
# path = '/home/scratch/glabarbera'
path = '/tsi/clusterhome/glabarbera/unet3d'
# path = '/media/glabarbera/Donnees'
#path = 'E:/DeePRAC_PROJECT/DatabaseDEEPRAC/nnunet'
task = 'Task208_NECKER'
dirpath = Path(path + '/nnUNet_preprocessed/' + task + '/nnUNetData_plans_v2.1_stage1')
#dirpath = Path(path + '/nnUNet_preprocessed/'+'/nnUNetData_plans_v2.1_2D_stage0')
#dirpath = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/imagesTs'
#labelpath = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/labelsTs'

folder = path + '/nnUNet_preprocessed/' + task + '/nnUNetData_plans_v2.1_stage1/Patches_perp_25'
# folder = path  +'/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_2D_stage0/Slices'

if not os.path.exists(folder):
    os.mkdir(folder)

#(_, _, filenames_imts) = next(os.walk(labelpath))
#(_, _, filenames_its) = next(os.walk(dirpath))
#filenames_imts = sorted(filenames_imts)
#filenames_its = sorted(filenames_its)
import time
start_time = time.time()
for name in dirpath.iterdir():
    if (name.is_file()) & (name.suffix == '.npz'):
        print(str(name)[-19:-4])
        if os.path.exists(folder + '/' + str(name)[-19:-4] + '_1_1_0.npz'):
            continue
        img = np.load(name)['data'][0, :, :, :].astype(np.float32)
        seg_r = np.load(name)['data'][1, :, :, :].astype(np.float32)

#for i in range(-1,-len(filenames_imts),-1):
    #spath = os.path.join(labelpath, filenames_imts[i])
    #ipath = os.path.join(dirpath, filenames_its[i])
    #print(spath )
    #seg, affine,_ = load_nii(spath)
    #img_, imgaffine, hdr = load_nii(ipath)

    #img = resample_nii(img_, hdr, [0.45703101, 0.45703101, 0.8999939])
    #seg_r = resample_nii(seg, hdr, [0.45703101, 0.45703101, 0.8999939],mode='target')

    #save_nii_norescale(img, imgaffine, 'C:/Users/Giammarco/Desktop/img.nii.gz')
    #save_nii(seg_r,affine,'C:/Users/Giammarco/Desktop/seg.nii.gz')

    #seg_r,affine,_ = load_nii('C:/Users/Giammarco/Desktop/seg.nii.gz')
    #img, imgaffine, hdr = load_nii('C:/Users/Giammarco/Desktop/img.nii.gz')
    #preprocessing = [188.93, 175.44, 1034.5, -3.0]
    #img = rescaled(img,preprocessing)

        print(seg_r.shape)
        print(np.unique(seg_r))
        n_struct = seg_r.max().astype(int)
        print(img.shape)
        print(img.max(),img.min())

        # 0. BIFURCATION POINTS AND ISLANDS
        #seg_r = seg_r// np.unique(seg_r)[1]
        #print(np.unique(seg_r))
        for struct in range(1,n_struct+1):
            print(struct)
            dat = seg_r == struct
            skel = skeletonize(dat)
            skel = skel / skel.max()
            bif = bif_conv(skel)
            time0 = time.time()
            print("bifurcations --- %s seconds ---" % (time0 - start_time))
            #save_nii(bif, affine, 'C:/Users/Giammarco/Desktop/bif.nii.gz')
            skel[np.nonzero(bif)]=0
            newskel, N = cc3d.connected_components(skel.astype(int), connectivity=26, return_N=True)
            print("The total number of islands is", N)
            time1 = time.time()
            print("islands --- %s seconds ---" % (time1 - time0))
            #save_nii(newskel, affine, 'C:/Users/Giammarco/Desktop/islands.nii.gz')
            voxel = combinator(*newskel.shape)

            for m in range(1,N+1):
                print('island n:', m)
                itemindex = np.argwhere(newskel==m)
                n_points = len(itemindex)
                print("The total number of points is", n_points)
                if n_points<2:
                    continue
                p0 = itemindex[0]
                p1 = itemindex[-1]
                a,b,c = direction(p0,p1)
                c0 = np.array([a, b, c])
                points, planes = [],[]
                patches = np.full((n_points,2,3,64,64),img.min(), dtype=np.float32)
                patches[:,1,:,:,:] = 0
                for q in range(n_points):
                    print('point n:', q)
                    #1. PLANE EXTRACTION
                    zero = time.time()
                    p = itemindex[q]
                    points.append(p)
                    d = plane(p,a,b,c)
                    p_in_plane = np.full((newskel.shape),-1)
                    op = voxel[:, 0] * c0[0] + voxel[:, 1] * c0[1] + voxel[:, 2] * c0[2] + d
                    p_in_plane[tuple((voxel[np.where(op==0)].astype(int)).T)] = 1
                    one = time.time()
                    print("plane extraction --- %s seconds ---" % (one - zero))
                    #save_nii(p_in_plane,affine,'C:/Users/Giammarco/Desktop/plane.nii.gz')

                    #2. PROJECTION OF THE PLANE
                    piano = np.argwhere(p_in_plane == 1)
                    pianomatrix = tuple(piano.T)
                    perp = np.argmax(abs(c0))
                    choicelist = np.array([0,1,2])
                    dimension = choicelist[np.where(choicelist!=perp)]

                    msp,segmsp = np.full((img.shape[dimension[0]], img.shape[dimension[1]]),img.min()), np.full((seg_r.shape[dimension[0]], seg_r.shape[dimension[1]]),-1)
                    msp[pianomatrix[dimension[0]],pianomatrix[dimension[1]]] = img[pianomatrix]
                    segmsp[pianomatrix[dimension[0]],pianomatrix[dimension[1]]] = seg_r[pianomatrix]
                    point = [p[dimension[0]], p[dimension[1]]]
                    two = time.time()
                    print("plane projection --- %s seconds ---" % (two - one))

                    #3. FORCING PLANE (APPROXIMATION)
                    for j in range(0, img.shape[dimension[0]]):
                        for k in range(0, img.shape[dimension[1]]):
                            if segmsp[j,k] == -1:
                                i = forcing(perp,dimension,j,k,c0,d)
                                i = np.clip(i,0,img.shape[perp] - 1)
                                choicelist[perp], choicelist[dimension[0]], choicelist[dimension[1]] = i, j, k
                                #print(choicelist)
                                msp[j, k] = img[tuple(choicelist)]
                                segmsp[j, k] = seg_r[tuple(choicelist)]
                                p_in_plane[tuple(choicelist)] = 1
                    tre = time.time()
                    print("forcing plane --- %s seconds ---" % (tre - two))
                    #save_nii(p_in_plane, affine, 'C:/Users/Giammarco/Desktop/plane_app.nii.gz')

                    #4. PATCHES EXTRACTION
                    newpiano = np.argwhere(p_in_plane == 1)
                    newpianomatrix = tuple(newpiano.T)
                    planes.append(newpianomatrix)
                    patch = msp[point[0]-32:point[0]+32, point[1]-32:point[1]+32]
                    segpatch = segmsp[point[0]-32:point[0]+32, point[1]-32:point[1]+32]
                    patches[q, 0, 1, 0:patch.shape[0], 0:patch.shape[1]] = patch
                    patches[q, 1, 1, 0:patch.shape[0], 0:patch.shape[1]] = segpatch
                    qtr = time.time()
                    print("patches extraction --- %s seconds ---" % (qtr - tre))

                    beforepiano, afterpiano = np.copy(newpiano), np.copy(newpiano)
                    beforepiano[:,perp] = newpiano[:,perp] - 1
                    beforepiano[:,perp] = np.clip(beforepiano[:,perp],0,img.shape[perp] - 1)
                    afterpiano[:,perp] = newpiano[:,perp] + 1
                    afterpiano[:, perp] = np.clip(afterpiano[:, perp], 0, img.shape[perp] - 1)
                    beforepianomatrix = tuple(beforepiano.T)
                    afterpianomatrix = tuple(afterpiano.T)

                    msp[beforepianomatrix[dimension[0]], beforepianomatrix[dimension[1]]] = img[beforepianomatrix]
                    segmsp[beforepianomatrix[dimension[0]], beforepianomatrix[dimension[1]]] = seg_r[beforepianomatrix]
                    patch = msp[point[0]-32:point[0]+32, point[1]-32:point[1]+32]
                    segpatch = segmsp[point[0]-32:point[0]+32, point[1]-32:point[1]+32]
                    patches[q, 0, 0, 0:patch.shape[0], 0:patch.shape[1]] = patch
                    patches[q, 1, 0, 0:patch.shape[0], 0:patch.shape[1]] = segpatch

                    msp[afterpianomatrix[dimension[0]], afterpianomatrix[dimension[1]]] = img[afterpianomatrix]
                    segmsp[afterpianomatrix[dimension[0]], afterpianomatrix[dimension[1]]] = seg_r[afterpianomatrix]
                    patch = msp[point[0]-32:point[0]+32, point[1]-32:point[1]+32]
                    segpatch = segmsp[point[0]-32:point[0]+32, point[1]-32:point[1]+32]
                    patches[q, 0, 2, 0:patch.shape[0], 0:patch.shape[1]] = patch
                    patches[q, 1, 2, 0:patch.shape[0], 0:patch.shape[1]] = segpatch

                    five = time.time()
                    print("RGB extraction --- %s seconds ---" % (five - qtr))


                #for w in range(n_points//64 + 1):
                #    save_nii_norescale(patches[w,0,:,:,:], affine, 'C:/Users/Giammarco/Desktop/PATCHES_118/imagesTs/island_'+str(m)+'_'+str(w)+'.nii.gz')
                #    save_nii_norescale(patches[w,1,:,:,:], affine, 'C:/Users/Giammarco/Desktop/PATCHES_118/labelsTs/island_'+str(m)+'_'+str(w)+'.nii.gz')
                for z in range(n_points):
                    np.savez(folder + '/' + str(name)[-19:-4] + '_' + str(struct) + '_' + str(m) +'_'+ str(z) + '.npz', patches[z, :, :, :, :])

        print("EXECUTION TIME this subject --- %s seconds ---" % (time.time() - start_time))