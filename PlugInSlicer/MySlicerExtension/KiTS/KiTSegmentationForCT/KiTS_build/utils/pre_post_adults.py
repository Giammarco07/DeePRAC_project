import numpy as np
import os
from utils.pre_processing import load_nii, save_nii, resample_nii, remove_small_islands
from utils.patches import adjust_dim, prepare_patches, from_image_to_original_nii, creation_gauss_filter, from_image_to_original
from scipy.ndimage.measurements import label
from utils.cropping import crop_to_nonzero, adjust_to_bbox
from scipy.ndimage import median_filter

def preprocessing_a(orig_path,patch_size):
    print('starting pre processing version 3.0 for ', orig_path)

    orig_crop, affine, hdr = load_nii(orig_path)
    img_r, bbox = crop_to_nonzero(orig_crop,-300)
    img_rr = np.rollaxis(img_r, 2, 0)
    img_rrr = np.rollaxis(img_rr, 2, 1)

    print(hdr.get_zooms())
    print('image:' , img_r.shape)

    #img_r = resample_nii(img_crop, hdr, res, mode = 'image')
    #print('new resolution: ', res)
    #print('after resampling: ', img_r.shape)


    dpatch_size = patch_size[0]
    hpatch_size = patch_size[1]
    wpatch_size = patch_size[2]
    d = dpatch_size - (img_rrr.shape[0] % dpatch_size)
    h = hpatch_size - (img_rrr.shape[1] % hpatch_size)
    w = wpatch_size - (img_rrr.shape[2] % wpatch_size)
    doverlap_stepsize = int(round(dpatch_size / 2))
    hoverlap_stepsize = int(round(hpatch_size / 2))
    woverlap_stepsize = int(round(wpatch_size / 2))



    x_ = adjust_dim(img_rrr, img_rrr.shape[0] + d, img_rrr.shape[1] + h, img_rrr.shape[2] + w)
    patch_x_ = prepare_patches(x_, dpatch_size, hpatch_size, wpatch_size, doverlap_stepsize,
                                hoverlap_stepsize, woverlap_stepsize)
    print('DONE')


    utils_for_post = [affine, hdr, patch_x_, x_.shape, img_rrr.shape, orig_crop.shape, bbox]

    return len(patch_x_), utils_for_post, x_

def postprocessing_a(pred3d, data_results, name_orig, utils_for_post, net,x_):
    print('starting postprocessing for prediction...')
    name = name_orig

    affine, hdr, patch_ids, x_shape, rrshape, origshape, bbox = utils_for_post

    newpath = data_results + '/pred_' + name

    pred3d[1] = median_filter(pred3d[1], size = 3)
    pred3d[1][pred3d[1] > 0.4] = 1
    if net==4:
        pred3d[0][x_<300] = 1
    pred = np.argmax(pred3d, axis=0)

    s_image__ = np.rollaxis(pred, 2, 0)
    s_image = np.rollaxis(s_image__, 2, 1)
    s = from_image_to_original(s_image, rrshape[2], rrshape[1], rrshape[0])
    #s = from_image_to_original_nii(s_image, rrshape[2], rrshape[1], rrshape[0], cropshape[0], cropshape[1], cropshape[2])
    s = adjust_to_bbox(s,bbox,origshape[0],origshape[1],origshape[2])

    s_final = np.rint(s)
    print(np.unique(s_final))
    print(s_final.shape)


    new = np.zeros((s_final.shape))
    structure = np.ones((3, 3, 3), dtype=np.int)
    for i in range(1,int(s_final.max()+1)):
        str = s_final == i
        labeled, _ = label(str, structure)
        print(np.unique(labeled))
        labeled_new, _ = remove_small_islands(labeled, 1000)
        print(np.unique(labeled))
        new[labeled_new>0] = i
    ncs = 0

    np.save(data_results + '/ncs.npy',ncs)
    save_nii(np.asarray(new, dtype=np.int8), affine, newpath)
    print(name, 'CORRECTLY SAVED IN /PRED')