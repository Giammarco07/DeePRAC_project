from pathlib import Path
import os
import numpy as np
from skimage.morphology import skeletonize
from utils.patches import create_patches2, prepare_patches, adjust_dim
from utils.cropping import crop_to_bbox, crop_to_nonzero
from utils.pre_processing import load_nii,  rescaled
from natsort import natsorted
from skimage.filters import frangi
import random

def rescale(data):
    rescaled = ((data - data.min()) / (data.max() - data.min()))
    return rescaled

#path = '/home/infres/glabarbera/nnunet'
#path = '/home/scratch/glabarbera
path = '/tsi/clusterhome/glabarbera/unet3d'
#path = '/media/glabarbera/Donnees'
task = 'Task308_NECKER'
task1 = 'Task308_NECKER'
dirpath = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1_affine')
dirpath = [Path(p) for p in natsorted([str(p) for p in dirpath.iterdir()])]
#dirpath = Path(path + '/nnUNet_preprocessed/'+'/nnUNetData_plans_v2.1_2D_stage0')
#dirpath = Path(path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+'/imagesTr')
#labelpath = Path(path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+'/labelsTr')

folder = path + '/nnUNet_preprocessed/'+task1+'/nnUNetData_plans_v2.1_stage1_affine/Patches'
#folder = path  +'/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_2D_stage0/Slices'

if not os.path.exists(folder):
    os.mkdir(folder)

for k in range(1,10):
    if not os.path.exists(folder + '/fold' + str(k)):
        os.mkdir(folder + '/fold' + str(k))

#fold = 1
#cc = 1
for ii in range(0,len(dirpath),1):
    i = dirpath[ii]
    #i1 = dirpath[ii+1]
    if (i.is_file()) & (i.suffix == '.npz'):
            print(i)
            img = np.load(i) #['data'][:, :, :, :].astype(np.float32)
            image = img[img.files[0]][:, :, :, :].astype(np.float32)
            print(image.shape, image.min(), image.max())
            
            #print(i1)
            #img1 = np.load(i1) #['data'][:, :, :, :].astype(np.float32)
            #image1 = img1[img1.files[0]][:, :, :, :].astype(np.float32)
            #print(image1.shape, image1.min(), image1.max())

            #if str(i)[-8:-4]=='0000':
            #    image = np.rollaxis(image, 3, 2)
            #    print('adults new:', image.shape)
            #else:
            #    img_rr = np.rollaxis(image, 2, 1)
            #    img_rrr = np.rollaxis(img_rr, 3, 2)
            #    image = img_rrr
            #    print('children new:', image.shape)

            seg_crop, bbox = crop_to_nonzero(image[-1,:,:,:], 0)
            print(seg_crop.shape)
            img_crop = crop_to_bbox(image[0,:,:,:], bbox)
            ##img_crop1 = crop_to_bbox(image[1,:,:,:], bbox)
            ##img_crop1 = crop_to_bbox(image1[0,:,:,:], bbox)
            
            #img_crop, bbox = crop_to_nonzero(image[0,:,:,:], -300)
            #print(img_crop.shape)
            #seg_crop = crop_to_bbox(image[-1,:,:,:], bbox)

            ims = img_crop.shape
            print(img_crop.shape)
           
            
            #preprocessing = [[106.34, 104.48, 391.0, -69.0],[106.34, 104.48, 391.0, -69.0]]  # mean,sd,p95,p05
            #preprocessing = [[175.91, 100.48, 631.98, -12.0],[36.03, 43.69, 112.0, -98.40]]  # mean,sd,p95,p05
            
            #preprocessing = [[175.91, 100.48, 391.0, -69.0],[106.34, 104.48, 391.0, -69.0]]  # mean,sd,p95,p05
            
            #preprocessing = [[175.91, 100.48, 631.98, -12.0],[175.91, 100.48, 631.98, -12.0]]  # mean,sd,p95,p05
            #preprocessing = [[-39.0, 70.0, 303.0, -70.0],[-39.0, 70.0, 303.0, -70.0]]  # mean,sd,p95,p05
            #img_crop = rescaled(img_crop, preprocessing[0])
            #img_crop1 = rescaled(img_crop1, preprocessing[1])
            
            #img_crop1 += (img_crop-img_crop1)*0.5
            
            #img_c = np.copy(img_crop)
            #maximum, minimum = img_c.max(),img_c.min()
            #img_c[img_c>=0] +=  0.2*img_c[img_c>=0]
            #img_c[img_c<0] -= 0.2*img_c[img_c<0]
            #img_c = np.clip(img_c,minimum,maximum)
            #img_crop1 = frangi(rescale(img_c), scale_range=(3,8), scale_step=5/4, alpha=0.5, beta=0.5, gamma=15, black_ridges=False, mode='reflect', cval=0)
            #img_crop1[img_crop1>0]=1

	    
            #img_crop,bbox =  crop_to_nonzero(image[0,:,:,:], image[0,:,:,:].min() + 0.005)
            #seg_crop = crop_to_bbox(image[1,:,:,:],bbox)
            #ims = img_crop.shape
            #print(img_crop.shape)
            
            
            limit = -19
            print(str(i)[limit:-4])
            dpatch_size = 96
            hpatch_size = 160
            wpatch_size = 160
            doverlap_stepsize = 48
            hoverlap_stepsize = 80
            woverlap_stepsize = 80
            d = dpatch_size - (ims[0] % dpatch_size)
            h = hpatch_size - (ims[1] % hpatch_size)
            w = wpatch_size - (ims[2] % wpatch_size)
            
            image1 = adjust_dim(img_crop,ims[0] + d, ims[1] + h, ims[2] + w)

            #image11 = adjust_dim(img_crop1,ims[0] + d, ims[1] + h, ims[2] + w)

            image2 = adjust_dim(seg_crop,ims[0] + d, ims[1] + h, ims[2] + w)
            
            image3 = np.asarray([image1, image2], dtype=np.float32)
            #image31 = np.asarray([image11, image2], dtype=np.float32)
            #image3 = np.asarray([image1, image11, image2], dtype=np.float32)
            
            
            print('prepare patches')
            patch_ids = prepare_patches(image1, dpatch_size, hpatch_size, wpatch_size, doverlap_stepsize, hoverlap_stepsize, woverlap_stepsize)
            print('total patches: ', len(patch_ids))
            print('create patches')
            images = create_patches2(image3, patch_ids, dpatch_size, hpatch_size, wpatch_size,2)
            #images1 = create_patches2(image31, patch_ids, dpatch_size, hpatch_size, wpatch_size,2)
            print('save patches')
            print(images.shape)
            #print(images1.shape)
            
            #images11 = np.copy(images)          
            for z in range(len(patch_ids)):
                #for r in np.arange(0,1.1,0.1):
                 #images11[z,0,:,:,:] = images1[z,0,:,:,:] + (images[z,0,:,:,:]-images1[z,0,:,:,:])*r
                 np.savez_compressed(folder + '/' + str(i)[limit:-4] + '_' + str(z+100) + '.npz', images[z,:,:,:,:])
                 #np.savez_compressed(folder + '/' + str(i)[limit:-4] + '_' + str(z+1000) + '.npz', images1[z,:,:,:,:])
                 #np.savez_compressed(folder + '/' + str(i)[limit:-4] + '_' + str(z+100) + '_' + str(int(r*10)) + '.npz', images11[z,:,:,:,:])
                 #for p in range(1,10):
                  #if p!=(10-fold):
                   #np.savez_compressed(folder + '/fold' + str(p) + '/' + str(i)[limit:-4] + '_' + str(z+100) + '.npz', images[z,:,:,:,:])
                   #np.savez_compressed(folder + '/fold' + str(p) + '/' + str(i)[limit:-4] + '_' + str(z+1000) + '.npz', images1[z,:,:,:,:])
                   #np.savez_compressed(folder + '/fold' + str(p) + '/' + str(i)[limit:-4] + '_' + str(z+100) + '_' + str(int(r*10)) + '.npz', images11[z,:,:,:,:])
            
            #fold+=1            
            #if cc==2:
            #    fold+=1
            #    cc=1
            #else:
            #    cc=2
'''

            for k in range(img_crop.shape[0]):
                    data = seg_crop[k, :, :] > 0

                    skeleton = skeletonize(data)
                    coordinates_grid = np.ones((2, skeleton.shape[0], skeleton.shape[1]), dtype=np.int16)
                    coordinates_grid[0] = coordinates_grid[0] * np.array([range(skeleton.shape[0])]).T
                    coordinates_grid[1] = coordinates_grid[1] * np.array([range(skeleton.shape[0])])

                    mask = skeleton != 0
                    non_zero_coords = np.hstack((coordinates_grid[0][mask].reshape(-1, 1),
                                                 coordinates_grid[1][mask].reshape(-1, 1)))
                    for j in range(len(non_zero_coords)):
                            loc = non_zero_coords[j]
                            if k+1==img_crop.shape[0] or loc[0]-32<0 or loc[0]+32>=img_crop.shape[1] or loc[1]-32<0 or loc[1]+32>=img_crop.shape[2]:
                                print('out of bounds')
                                pass
                            else:
                                patches = np.asarray([img_crop[k-1:k+2,loc[0] - 32:loc[0] + 32, loc[1] - 32:loc[1] + 32], seg_crop[k-1:k+2,loc[0] - 32:loc[0] + 32, loc[1] - 32:loc[1] + 32]], dtype=np.float32)
                                np.savez(folder + '/' + str(i)[-19:-4] + '_' + str(j+100) + '.npz', patches)


(_, _, filenames_imts) = next(os.walk(dirpath))
(_, _, filenames_lbts) = next(os.walk(labelpath))
filenames_imts = sorted(filenames_imts)
filenames_lbts = sorted(filenames_lbts)
for i in range(len(filenames_imts)):
        img, _, _ = load_nii(os.path.join(dirpath, filenames_imts[i]))
        print(img.shape)
        seg, _, _ = load_nii(os.path.join(labelpath, filenames_lbts[i]))
        print(seg.shape)
        img_crop, bbox = crop_to_nonzero(img, -300)
        seg_crop = crop_to_bbox(seg, bbox)
        print(img_crop.shape)
        x_KK = np.rollaxis(img_crop, 2, 0)
        x_KKK = np.rollaxis(x_KK, 2, 1)
        x_KKK = ((x_KKK - x_KKK.min()) / (x_KKK.max() - x_KKK.min() + ee))
        y_KK = np.rollaxis(seg_crop, 2, 0)
        y_KKK = np.rollaxis(y_KK, 2, 1)
        print(x_KKK.shape)
        dpatch_size = 96
        hpatch_size = 160
        wpatch_size = 160
        doverlap_stepsize = 24
        hoverlap_stepsize = 40
        woverlap_stepsize = 40
        image1 = adjust_dim(x_KKK, dpatch_size, hpatch_size, wpatch_size)
        image2 = adjust_dim(y_KKK, dpatch_size, hpatch_size, wpatch_size)
        image3 = np.asarray([image1, image2], dtype=np.float32)
        print('prepare patches')
        patch_ids = prepare_patches(image1, dpatch_size, hpatch_size, wpatch_size, doverlap_stepsize, hoverlap_stepsize,
                                    woverlap_stepsize)
        print('total patches: ', len(patch_ids))
        print('create patches')
        images = create_patches2(image3, patch_ids, dpatch_size, hpatch_size, wpatch_size, 2)
        print('save patches')
        print(images.shape)

        for z in range(len(patch_ids)):
                np.savez_compressed(folder + '/' + filenames_imts[i][:-4] + '_' + str(z + 1000) + '.npz', images[z, :, :, :, :])
'''







