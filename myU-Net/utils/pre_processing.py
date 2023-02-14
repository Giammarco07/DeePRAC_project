import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage.transform import resize

def load_nii(path):
    img = nib.load(path)
    hdr = img.header
    #pix = hdr.get_zooms()
    #print(pix)
    data = img.get_fdata()
    #print(data.shape)
    affine = img.affine
    return data, affine, hdr

def save_nii(data, affine, path):
    rescaled = ((data - data.min())/ (data.max() - data.min()))*255.0
    res = rescaled.astype(np.uint8)
    img = nib.Nifti1Image(res, affine)
    nib.save(img, path)


def resample_nii(old_data,hdr,res, mode = 'image'):
    pix = hdr.get_zooms()
    pix_0 = max(pix)
    res_0 = max(res)
    print(pix_0,res_0)
    argres = np.argmax(res)
    if argres==0:
        if mode == 'image':
            if pix_0<(3*res_0):
                new_shape = ((int(round((pix[0]/res[0]) * old_data.shape[0])), int(round((pix[1]/res[1]) * old_data.shape[1])), int(round((pix[2]/res[2]) * old_data.shape[2]))))
                data = resize(old_data, new_shape, order=3, mode = 'constant', cval=old_data.min(), anti_aliasing = True)
                # we resample all cases to a common voxel spacing [mm] - cubic interpolation
            else:
                new_shape = ((old_data.shape[0], int(round((pix[1] / res[1]) * old_data.shape[1])),
                              int(round((pix[2] / res[2]) * old_data.shape[2]))))
                data = resize(old_data, new_shape, order=3, mode='constant', cval=old_data.min(), anti_aliasing=True)
                del old_data
                new_shape = ((int(round((pix[0] / res[0]) * data.shape[0])), data.shape[1], data.shape[2]))
                data = resize(data, new_shape, order=0, mode='constant', cval=data.min(), anti_aliasing=True,preserve_range=True)
        elif mode == 'image2D':
            new_shape = ((old_data.shape[0], int(round((pix[1] / res[1]) * old_data.shape[1])),
                          int(round((pix[2] / res[2]) * old_data.shape[2]))))
            data = resize(old_data, new_shape, order=3, mode='constant', cval=old_data.min(), anti_aliasing=True)
        elif mode == 'target':
                new_shape = ((int(round((pix[0] / res[0]) * old_data.shape[0])), int(round((pix[1] / res[1]) * old_data.shape[1])),
                              int(round((pix[2] / res[2]) * old_data.shape[2]))))
                data = resize(old_data, new_shape, order=0, mode='constant', cval=old_data.min(), clip=True, anti_aliasing=False,preserve_range=True)
        elif mode == 'target2D':
            new_shape = ((old_data.shape[0], int(round((pix[1] / res[1]) * old_data.shape[1])),
                          int(round((pix[2] / res[2]) * old_data.shape[2]))))
            data = resize(old_data, new_shape, order=0, mode='constant', cval=old_data.min(), anti_aliasing=False,preserve_range=True)

    elif argres==2:
        if mode == 'image':
            if pix_0<(3*res_0):
                new_shape = ((int(round((pix[0]/res[0]) * old_data.shape[0])), int(round((pix[1]/res[1]) * old_data.shape[1])), int(round((pix[2]/res[2]) * old_data.shape[2]))))
                data = resize(old_data, new_shape, order=3, mode = 'constant', cval=old_data.min(), anti_aliasing = True)
                # we resample all cases to a common voxel spacing [mm] - cubic interpolation
            else:
                new_shape = ((int(round((pix[0]/res[0]) * old_data.shape[0])), int(round((pix[1] / res[1]) * old_data.shape[1])),
                             old_data.shape[2]))
                data = resize(old_data, new_shape, order=3, mode='constant', cval=old_data.min(), anti_aliasing=True)
                del old_data
                new_shape = ((data.shape[0], data.shape[1], int(round((pix[2]/res[2]) * data.shape[2]))))
                data = resize(data, new_shape, order=0, mode='constant', cval=data.min(), anti_aliasing=True,preserve_range=True)
        elif mode == 'image2D':
            new_shape = ((int(round((pix[0] / res[0]) * old_data.shape[0])), int(round((pix[1] / res[1]) * old_data.shape[1])),
                          old_data.shape[2]))
            data = resize(old_data, new_shape, order=3, mode='constant', cval=old_data.min(), anti_aliasing=True)
        elif mode == 'target':
                new_shape = ((int(round((pix[0]/res[0]) * old_data.shape[0])), int(round((pix[1]/res[1]) * old_data.shape[1])), int(round((pix[2]/res[2]) * old_data.shape[2]))))
                data = resize(old_data, new_shape, order=0, mode = 'constant', cval=old_data.min(), anti_aliasing = False,preserve_range=True)
                # we resample all cases to a common voxel spacing [mm] - bilinear interpolation
        elif mode == 'target2D':
            new_shape = ((int(round((pix[0] / res[0]) * old_data.shape[0])), int(round((pix[1] / res[1]) * old_data.shape[1])),
                          old_data.shape[2]))
            data = resize(old_data, new_shape, order=0, mode='constant', cval=old_data.min(), anti_aliasing=False,preserve_range=True)

    else:
        data = old_data
        print('Warning! Depth should be in position 0 or 2 in res.')

    return data

def resample_nrrd(old_data,hdr,res, mode = 'image'):
    space = hdr['space directions']
    pix = [space[2,2], space[1,1], space[0,0]]
    if mode == 'image':
        if pix[0]<res[0]:
            data = zoom(old_data, (round(pix[0]/res[0], 3), round(pix[1]/res[1], 3), round(pix[2]/res[2], 3)), order=3)
            # we resample all cases to a common voxel spacing [mm] - trilinear interpolation
        else:
            data = zoom(old_data, (1, round(pix[1] / res[1], 3), round(pix[2] / res[2], 3)),order=3)
            data = zoom(data, (round(pix[0] / res[0], 3), 1, 1), order=0)
    elif mode == 'target':
        if pix[0]<res[0]:
            data = zoom(old_data, (round(pix[0]/res[0], 3), round(pix[1]/res[1], 3), round(pix[2]/res[2], 3)), order=1)
            # we resample all cases to a common voxel spacing [mm] - trilinear interpolation
        else:
            data = zoom(old_data, (1, round(pix[1] / res[1], 3), round(pix[2] / res[2], 3)),order=1)
            data = zoom(data, (round(pix[0] / res[0], 3), 1, 1), order=0)
        data[data <= 0.5] = 0.0
        data[data > 0.5] = 1.0

    return data

def rescaled(data,preprocessing):
    np.clip(data, preprocessing[3], preprocessing[2], out=data)
    data = (data - preprocessing[0]) / preprocessing[1]   # data - mean / std_dev of foreground
    #data = ((data - data.min()) / (data.max() - data.min()))*1.0

    return data

def rescaled_ad(data):
    np.clip(data, -78.0, 303.0, out=data)
    data = (data - 99.9) / 77.1   # data - mean / std_dev of foreground
    #data = ((data - data.min()) / (data.max() - data.min()))*1.0

    return data

def seg_label(seg, channel_dim):
    labels= np.zeros((seg.shape[0],seg.shape[1],seg.shape[2], (channel_dim-1)))

    label_s1 = np.zeros(seg.shape, dtype=np.uint8)
    label_s1[np.where(seg == 1)] = 1
    labels[:, :, :, 0] = label_s1
    del label_s1

    if channel_dim>2:
        label_s2 = np.zeros(seg.shape, dtype=np.uint8)
        label_s2[np.where(seg == 2)] = 1
        labels[:, :, :, 1] = label_s2
        del label_s2
    if channel_dim>3:
        label_s3 = np.zeros(seg.shape, dtype=np.uint8)
        label_s3[np.where(seg == 3)] = 1
        labels[:, :, :, 2] = label_s3
        del label_s3
    if channel_dim>4:
        label_s4 = np.zeros(seg.shape, dtype=np.uint8)
        label_s4[np.where(seg == 4)] = 1
        labels[:, :, :, 3] = label_s4
        del label_s4


    return np.asarray(labels, dtype=np.uint8)

def seg_label_children(seg):
    seg[np.where(seg ==1)] = 0 #kidney_s
    seg[np.where(seg ==2)] = 0 #kidney_r
    seg[np.where(seg ==3)] = 0 #cyst
    seg[np.where(seg ==4)] = 0 #tumor_s
    seg[np.where(seg ==5)] = 0 #tumor_r
    seg[np.where(seg == 6)] = 0  #ureters
    seg[np.where(seg == 7)] = 1  #arteries
    seg[np.where(seg == 8)] = 2 #veins

    return np.asarray(seg, dtype=np.int8)

def table_children(age,loc):
    group = 0
    if int(age) <= 2 and loc == 'R':
        group = 1
    elif int(age) <= 2 and loc =='L':
        group = 2
    elif int(age) <= 2 and loc =='B':
        group = 3
    elif (int(age) > 2 and int(age)<= 5) and loc == 'R':
        group = 4
    elif (int(age) > 2 and int(age)<= 5) and loc =='L':
        group = 5
    elif (int(age) > 2 and int(age)<= 5) and loc =='B':
        group = 6
    if int(age) > 5 and loc == 'R':
        group = 7
    elif int(age) > 5 and loc =='L':
        group = 8
    elif int(age) > 5 and loc =='B':
        group = 9
    return group
