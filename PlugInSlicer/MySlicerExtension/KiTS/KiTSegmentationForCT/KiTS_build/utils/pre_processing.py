import numpy as np
import nibabel as nib
from skimage.transform import resize

def load_nii(path):
    img = nib.load(path)
    hdr = img.header
    data = img.get_fdata()
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
    argres = np.argmax(res)
    if argres==0:
        if mode == 'image':
            if pix_0<(3*res_0):
                new_shape = ((int(round((pix[0]/res[0]) * old_data.shape[0])), int(round((pix[1]/res[1]) * old_data.shape[1])), int(round((pix[2]/res[2]) * old_data.shape[2]))))
                data = resize(old_data, new_shape, order=3, mode = 'constant', cval=old_data.min(), anti_aliasing = False)
                # we resample all cases to a common voxel spacing [mm] - cubic interpolation
            else:
                new_shape = ((old_data.shape[0], int(round((pix[1] / res[1]) * old_data.shape[1])),
                              int(round((pix[2] / res[2]) * old_data.shape[2]))))
                data = resize(old_data, new_shape, order=3, mode='constant', cval=old_data.min(), anti_aliasing=False)
                new_shape = ((int(round((pix[0] / res[0]) * data.shape[0])), data.shape[1], data.shape[2]))
                data = resize(data, new_shape, order=0, mode='constant', cval=old_data.min(), anti_aliasing=False)
        elif mode == 'target':
            if pix_0<(3*res_0):
                new_shape = ((int(round((pix[0] / res[0]) * old_data.shape[0])), int(round((pix[1] / res[1]) * old_data.shape[1])),
                              int(round((pix[2] / res[2]) * old_data.shape[2]))))
                data = resize(old_data, new_shape, order=1, mode='constant', cval=old_data.min(), clip=True, anti_aliasing=False)
                # we resample all cases to a common voxel spacing [mm] - bilinear interpolation
            else:
                new_shape = ((old_data.shape[0], int(round((pix[1] / res[1]) * old_data.shape[1])),
                              int(round((pix[2] / res[2]) * old_data.shape[2]))))
                data = resize(old_data, new_shape, order=1, mode='constant', cval=old_data.min(), clip=True, anti_aliasing=False)
                new_shape = ((int(round((pix[0] / res[0]) * data.shape[0])), data.shape[1], data.shape[2]))
                data = resize(data, new_shape, order=0, mode='constant', cval=old_data.min(), clip=True, anti_aliasing=False)
            data[data <= 0.5] = 0.0
            data[data > 0.5] = 1.0

    elif argres==2:
        if mode == 'image':
            if pix_0<(3*res_0):
                new_shape = ((int(round((pix[0]/res[0]) * old_data.shape[0])), int(round((pix[1]/res[1]) * old_data.shape[1])), int((round(pix[2]/res[2]) * old_data.shape[2]))))
                data = resize(old_data, new_shape, order=3, mode = 'constant', cval=old_data.min(), anti_aliasing = False)
                # we resample all cases to a common voxel spacing [mm] - cubic interpolation
            else:
                new_shape = ((int(round((pix[0]/res[0]) * old_data.shape[0])), int(round((pix[1] / res[1]) * old_data.shape[1])),
                             old_data.shape[2]))
                data = resize(old_data, new_shape, order=3, mode='constant', cval=old_data.min(), anti_aliasing=False)
                new_shape = ((data.shape[0], data.shape[1], int(round((pix[2]/res[2]) * old_data.shape[2]))))
                data = resize(data, new_shape, order=0, mode='constant', cval=old_data.min(), anti_aliasing=False)
        elif mode == 'target':
            if pix_0<(3*res_0):
                new_shape = ((int(round((pix[0]/res[0]) * old_data.shape[0])), int(round((pix[1]/res[1]) * old_data.shape[1])), int(round((pix[2]/res[2]) * old_data.shape[2]))))
                data = resize(old_data, new_shape, order=1, mode = 'constant', cval=old_data.min(), anti_aliasing = False)
                # we resample all cases to a common voxel spacing [mm] - bilinear interpolation
            else:
                new_shape = ((int(round((pix[0]/res[0]) * old_data.shape[0])), int(round((pix[1] / res[1]) * old_data.shape[1])),
                             old_data.shape[2]))
                data = resize(old_data, new_shape, order=1, mode='constant', cval=old_data.min(),clip=True, anti_aliasing=False)
                new_shape = ((data.shape[0], data.shape[1], int(round((pix[2]/res[2]) * old_data.shape[2]))))
                data = resize(data, new_shape, order=0, mode='constant', cval=old_data.min(),clip=True, anti_aliasing=False)
            data[data <= 0.5] = 0.0
            data[data > 0.5] = 1.0

    else:
        data = old_data
        print('Warning! Depth should be in position 0 or 2 in res.')

    return data


def rescaled(data,preprocessing):
    np.clip(data, preprocessing[3], preprocessing[2], out=data)
    data = (data - preprocessing[0]) / preprocessing[1]   # data - mean / std_dev of foreground

    return data

def rescaled_ad(data):
    np.clip(data, -78.0, 303.0, out=data)
    data = (data - 99.9) / 77.1   # data - mean / std_dev of foreground

    return data

def remove_small_islands(image, pix):
    count = 0
    for i in range(int(image.max())+1):
        str = image == i
        c = np.count_nonzero(str)
        if c<pix:
            image[str] = 0
            count += 1
    return image, count