import numpy as np
from skimage.transform import resize

def adjust_dim(image, dpatch_size, hpatch_size, wpatch_size, minn = None):
    if minn==None:
        minimum = image.min()
    else:
        minimum = minn
    D, H, W = image.shape
    flag = 0
    if D<dpatch_size:
        x=dpatch_size-D
        a=np.zeros((x,H,W))
        a[:,:,:]=minimum
        if x>1:
            xx = int(round(x/2))
            new_image = np.concatenate((a[0:xx,:,:], image, a[xx:x, :, :]), axis=0)
        else:
            new_image=np.concatenate((image,a), axis=0)
        del image
        flag = 1
    if H<hpatch_size:
        y=hpatch_size-H
        if flag == 1:
            D, H, W = new_image.shape
            b = np.zeros((D, y, W))
            b[:,:,:]=minimum
            if y > 1:
                yy = int(round(y / 2))
                new_image2 = np.concatenate((b[:,0:yy,:], new_image, b[:,yy:y,:]), axis=1)
            else:
                new_image2 = np.concatenate((new_image, b), axis=1)
            del new_image
        else:
            b = np.zeros((D, y, W))
            b[:,:,:]=minimum
            if y > 1:
                yy = int(round(y / 2))
                new_image2 = np.concatenate((b[:, 0:yy, :], image, b[:, yy:y, :]), axis=1)
            else:
                new_image2 = np.concatenate((image, b), axis=1)
            del image
        flag = 2
    if W<wpatch_size:
        z=wpatch_size-W
        if flag == 1:
            D, H, W = new_image.shape
            c = np.zeros((D, H, z))
            c[:,:,:]=minimum
            if z > 1:
                zz = int(round(z / 2))
                new_image3 = np.concatenate((c[:,:,0:zz], new_image, c[:,:, zz:z]), axis=2)
            else:
                new_image3 = np.concatenate((new_image, c), axis=2)
            del new_image
        elif flag == 2:
            D, H, W = new_image2.shape
            c = np.zeros((D, H, z))
            c[:,:,:]=minimum
            if z > 1:
                zz = int(round(z / 2))
                new_image3 = np.concatenate((c[:, :, 0:zz], new_image2, c[:, :, zz:z]), axis=2)
            else:
                new_image3 = np.concatenate((new_image2, c), axis=2)
            del new_image2
        else:
            c=np.zeros((D,H,z))
            c[:,:,:]=minimum
            if z > 1:
                zz = int(round(z / 2))
                new_image3 = np.concatenate((c[:, :, 0:zz], image, c[:, :, zz:z]), axis=2)
            else:
                new_image3 = np.concatenate((image, c), axis=2)
            del image
        flag = 3


    #centrare l'immagine rispetto depth:
    if flag==1:
        final_image = new_image
    elif flag==2:
        final_image = new_image2
    elif flag == 3:
        final_image = new_image3
    else:
        final_image = image

    return final_image


def prepare_patches(image, dpatch_size, hpatch_size, wpatch_size, doverlap_stepsize, hoverlap_stepsize, woverlap_stepsize):
	patch_ids = []

	D, H, W = image.shape

	drange = list(range(0, D-dpatch_size+1, doverlap_stepsize))
	hrange = list(range(0, H-hpatch_size+1, hoverlap_stepsize))
	wrange = list(range(0, W-wpatch_size+1, woverlap_stepsize))


	for d in drange:
		for h in hrange:
			for w in wrange:
				patch_ids.append((d, h, w))

	return patch_ids

def creation_gauss_filter(imgshape, patch_ids, dpatch_size, hpatch_size, wpatch_size):
    gauss = np.zeros((imgshape), dtype=float)
    gaussian_filter = gaussian_map((dpatch_size, hpatch_size, wpatch_size))
    for i in range(len(patch_ids)):
        (d, h, w) = patch_ids[i]
        gauss[d:d + dpatch_size, h:h + hpatch_size, w:w + wpatch_size] += gaussian_filter

    return gauss +1e-20

def creation_patch_filter(imgshape, patch_ids, dpatch_size, hpatch_size, wpatch_size):
    gauss = np.zeros((imgshape), dtype=float)
    gaussian_filter = np.ones((dpatch_size, hpatch_size, wpatch_size))
    for i in range(len(patch_ids)):
        (d, h, w) = patch_ids[i]
        gauss[d:d + dpatch_size, h:h + hpatch_size, w:w + wpatch_size] += gaussian_filter

    return gauss +1e-20

def from_image_to_original_nii(image, wsize, hsize, dsize, wshape, hshape, dshape):
    old_data = np.zeros((wsize, hsize, dsize))
    a = int(round((image.shape[0]-wsize)/2))
    b = int(round((image.shape[1]-hsize)/2))
    c = int(round((image.shape[2]-dsize)/2))
    old_data[:,:,:]=image[a:(wsize+a), b:(hsize+b), c:(dsize+c)]
    new_shape = ((wshape, hshape, dshape))
    data = resize(old_data, new_shape, order=0, mode = 'constant', cval=old_data.min(), anti_aliasing = False)

    return data

def from_image_to_original(image, wsize, hsize, dsize):
    old_data = np.zeros((wsize, hsize, dsize))
    a = int(round((image.shape[0]-wsize)/2))
    b = int(round((image.shape[1]-hsize)/2))
    c = int(round((image.shape[2]-dsize)/2))
    old_data[:,:,:]=image[a:(wsize+a), b:(hsize+b), c:(dsize+c)]

    return old_data

def gaussian_map(patch_size):
    # Creatian of the gaussian filter
    from scipy.ndimage.filters import gaussian_filter
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigma_scale = 1. / 4
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map