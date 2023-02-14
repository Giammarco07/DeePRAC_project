import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter

def adjust_dim(image, dpatch_size, hpatch_size, wpatch_size):
    minimum = image.min()
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


def adjust_dim_2d(image, dpatch_size, hpatch_size):
    minimum = image.min()
    D, H = image.shape
    flag = 0
    if D < dpatch_size:
        x = dpatch_size - D
        a = np.zeros((x, H))
        a[:, :] = minimum
        if x > 1:
            xx = round(x / 2)
            new_image = np.concatenate((a[0:xx, :], image, a[xx:x, :]), axis=0)
        else:
            new_image = np.concatenate((image, a), axis=0)
        del image
        flag = 1
    if H < hpatch_size:
        y = hpatch_size - H
        if flag == 1:
            D, H = new_image.shape
            b = np.zeros((D, y))
            b[:, :] = minimum
            if y > 1:
                yy = round(y / 2)
                new_image2 = np.concatenate((b[:, 0:yy], new_image, b[:, yy:y]), axis=1)
            else:
                new_image2 = np.concatenate((new_image, b), axis=1)
            del new_image
        else:
            b = np.zeros((D, y))
            b[:, :] = minimum
            if y > 1:
                yy = round(y / 2)
                new_image2 = np.concatenate((b[:, 0:yy], image, b[:, yy:y]), axis=1)
            else:
                new_image2 = np.concatenate((image, b), axis=1)
            del image
        flag = 2

    # centrare l'immagine rispetto depth:
    if flag == 1:
        final_image = new_image
    elif flag == 2:
        final_image = new_image2
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

def prepare_patches2d(image, hpatch_size, wpatch_size, hoverlap_stepsize, woverlap_stepsize):
	patch_ids = []

	_, H, W = image.shape

	hrange = list(range(0, H-hpatch_size+1, hoverlap_stepsize))
	wrange = list(range(0, W-wpatch_size+1, woverlap_stepsize))

	if (H-hpatch_size) % hoverlap_stepsize != 0:
		hrange.append(H-hpatch_size)
	if (W-wpatch_size) % woverlap_stepsize != 0:
		wrange.append(W-wpatch_size)

		for h in hrange:
			for w in wrange:
				patch_ids.append((h, w))

	return patch_ids


def creation_gauss_filter(imgshape, patch_ids, dpatch_size, hpatch_size, wpatch_size):
    gauss = np.zeros((imgshape[-3], imgshape[-2], imgshape[-1]), dtype=np.float32)
    gaussian_filter = gaussian_map((dpatch_size, hpatch_size, wpatch_size))
    for i in range(len(patch_ids)):
        (d, h, w) = patch_ids[i]
        gauss[d:d + dpatch_size, h:h + hpatch_size, w:w + wpatch_size] += gaussian_filter

    return gauss+1e-20

def creation_gauss_filter_skel(imgshape, patch_ids, dpatch_size, hpatch_size, wpatch_size):
    gauss = np.zeros((imgshape[-3], imgshape[-2], imgshape[-1]), dtype=np.float32)
    gaussian_filter = gaussian_map((dpatch_size, hpatch_size, wpatch_size))
    for i in range(len(patch_ids)):
        (d, h, w) = patch_ids[i]
        patch =  gauss[d-16:d+16,h-32:h+32,w-32:w+32]
        x,y,z = 16-patch.shape[0]//2, 32-patch.shape[1]//2, 32-patch.shape[2]//2
        gauss[d-16:d+16,h-32:h+32,w-32:w+32] += gaussian_filter[x:x+patch.shape[0],y:y+patch.shape[1],z:z+patch.shape[2]]

    return gauss+1e-20
        
def from_image_to_original_nii(hdr, image, wsize, hsize, dsize, wshape, hshape, dshape, res,mode=3):
    old_data = np.zeros((wsize, hsize, dsize))
    a = int(round((image.shape[0]-wsize)/2))
    b = int(round((image.shape[1]-hsize)/2))
    c = int(round((image.shape[2]-dsize)/2))
    old_data[:,:,:]=image[a:(wsize+a), b:(hsize+b), c:(dsize+c)]
    pix = hdr.get_zooms()
    argres = np.argmax(res)
    new_shape = ((wshape, hshape, dshape))
    '''
    if argres == 0:
        if mode==2:
            new_shape = ((old_data.shape[0], int(round((res[1] / pix[1]) * old_data.shape[1])),
                          int(round((res[2] / pix[2]) * old_data.shape[2]))))
        else:
            new_shape = ((int(round((res[0] / pix[0]) * old_data.shape[0])), int(round((res[1] / pix[1]) * old_data.shape[1])),
                          int(round((res[2] / pix[2]) * old_data.shape[2]))))
    else:
        if mode==2:
            new_shape = ((int(round((res[0] / pix[0]) * old_data.shape[0])), int(round((res[1] / pix[1]) * old_data.shape[1])),
                          old_data.shape[2]))
        else:
            new_shape = ((int(round((res[0] / pix[0]) * old_data.shape[0])), int(round((res[1] / pix[1]) * old_data.shape[1])),
                          int(round((res[2] / pix[2]) * old_data.shape[2]))))
    '''
    data = resize(old_data, new_shape, order=0, mode='constant', cval=old_data.min(), anti_aliasing=False,preserve_range=True)

    return data

def from_image_to_original_nrrd(hdr, image, dsize, hsize, wsize, res):
    new = np.zeros((dsize,hsize,wsize))
    a = round((image.shape[0]-dsize)/2)
    b = round((image.shape[1]-hsize)/2)
    c = round((image.shape[2]-wsize)/2)
    new[:,:,:]=image[a:(dsize+a), b:(hsize+b), c:(wsize+c)]
    space = hdr['space directions']
    pix = [space[2, 2], space[1, 1], space[0, 0]]
    if pix[0] < res[0]:
        original_image = zoom(new, (round(res[0] / pix[0], 3), round(res[1] / pix[1], 3), round(res[2] / pix[2], 3)),
                              order=3)
        # we resample all cases to a common voxel spacing [mm] - trilinear interpolation
    else:
        original_image = zoom(new, (1, round(res[1] / pix[1], 3), round(res[2] / pix[2], 3)), order=3)
        original_image = zoom(original_image, (round(res[0] / pix[0], 3), 1, 1), order=0)
    del new

    return original_image

def gaussian_map(patch_size):
    # Creatian of the gaussian filter
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigma_scale = 1. / 2 #8
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map
    
def gaussian_map_ddt(patch_size, zu):
    tmp = np.ones(patch_size)
    sigma = (zu/3)
    gaussian_importance_map = gaussian_filter(tmp, sigma, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    return gaussian_importance_map

def create_patches(image, patch_ids, dpatch_size, hpatch_size, wpatch_size):

    images = np.array([])
    for i in range(len(patch_ids)):
        (d, h, w) = patch_ids[i]
        _image = image[d:d + dpatch_size, h:h + hpatch_size, w:w + wpatch_size]
        images = np.append(images, _image)
        del _image

    return images

def create_patches2(image, patch_ids, dpatch_size, hpatch_size, wpatch_size,channel_dim):

    images = np.zeros((len(patch_ids),channel_dim,dpatch_size, hpatch_size, wpatch_size))
    for i in range(len(patch_ids)):
        (d, h, w) = patch_ids[i]
        images[i,:,:,:,:] = image[:, d:d + dpatch_size, h:h + hpatch_size, w:w + wpatch_size]

    return images
