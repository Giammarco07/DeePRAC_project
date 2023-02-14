import numpy as np
from utils.pre_processing import load_nii, save_nii, resample_nii, remove_small_islands
from utils.patches import adjust_dim, prepare_patches, from_image_to_original_nii, creation_gauss_filter, creation_patch_filter, from_image_to_original
from scipy.ndimage.measurements import label
from skimage.filters import hessian
from utils.cropping import crop_to_nonzero, adjust_to_bbox
import nibabel as nib
from scipy.ndimage import median_filter


def preprocessing_c(img_path,orig_path, patch_size_pre):
    print('starting pre processing version 3.0 for ', img_path)


    img_crop, affine, hdr = load_nii(img_path)
    img_orig, _, _ = load_nii(orig_path)

    print(hdr.get_zooms())
    print('after crop:' , img_crop.shape)
    img_r = img_crop
    #img_r = resample_nii(img_crop, hdr, res, mode = 'image')
    #print('new resolution: ', res)
    #print('after resampling: ', img_r.shape)
    img_rr = np.rollaxis(img_r, 2, 0)
    img_rrr = np.rollaxis(img_rr, 2, 1)

    patch_size = patch_size_pre
    dpatch_size = patch_size[0]
    hpatch_size = patch_size[1]
    wpatch_size = patch_size[2]
    d = dpatch_size - (img_rrr.shape[0] % dpatch_size)
    h = hpatch_size - (img_rrr.shape[1] % hpatch_size)
    w = wpatch_size - (img_rrr.shape[2] % wpatch_size)
    doverlap_stepsize = int(round(dpatch_size / 2))
    hoverlap_stepsize = int(round(hpatch_size / 2))
    woverlap_stepsize = int(round(wpatch_size / 2))

    x_0 = adjust_dim(img_rrr, img_rrr.shape[0] + d, img_rrr.shape[1] + h, img_rrr.shape[2] + w, minn=img_orig.min())
    patch_x_0 = prepare_patches(x_0, dpatch_size, hpatch_size, wpatch_size, doverlap_stepsize,
                                               hoverlap_stepsize, woverlap_stepsize)

    utils_for_post = [affine, hdr, patch_x_0, x_0.shape,img_rrr.shape, img_crop.shape]

    return len(patch_x_0), utils_for_post, x_0

def postprocessing_c(pred3d, data_results, name_orig, utils_for_post, channel_dim,net):
    print('starting postprocessing for prediction...')
    name = name_orig

    affine, hdr, patch_ids0, x_shape0, rrshape, cropshape = utils_for_post


    newpath = data_results + '/pred_' + name


    '''
    if net>1:
        print('applying vesselness...')
        if net == 3:
            best_sigmamin = 3.0
            best_sigmamax = 5.0
        else:
            best_sigmamin = 1.0
            best_sigmamax = 3.0

        for k in range(1,channel_dim):
            pred3d[k] = median_filter(pred3d[k], size=3)
            hseg = 1 - hessian(pred3d[k], scale_range=(best_sigmamin, best_sigmamax),
                               scale_step=(best_sigmamax - best_sigmamin) / 4,
                               alpha=0.5, beta=0.5, gamma=15, black_ridges=False, mode='reflect', cval=0)
            pred3d[k] *= hseg
            pred3d[k][pred3d[k]>0.4] = 1
        print('applying vesselness...DONE')
    else:
    '''
    for k in range(1, channel_dim):
        pred3d[k] = median_filter(pred3d[k], size = 3)
        pred3d[k][pred3d[k] > 0.4] = 1


    pred = np.argmax(pred3d, axis=0)
    s_image__ = np.rollaxis(pred, 2, 0)
    s_image = np.rollaxis(s_image__, 2, 1)
    s = from_image_to_original(s_image, rrshape[2], rrshape[1], rrshape[0])
    #s = from_image_to_original_nii(s_image, rrshape[2], rrshape[1], rrshape[0], cropshape[0], cropshape[1], cropshape[2])


    s_final = np.rint(s)
    print(np.unique(s_final))
    print(s_final.shape)


    print('removing small structures')
    if net!=3:
        new = np.zeros((s_final.shape))
        ncs = np.zeros((int(s_final.max())))
        structure = np.ones((3, 3, 3), dtype=np.int)
        for i in range(1,int(s_final.max()+1)):
            strr = s_final == i
            labeled, ncomponents = label(strr, structure)
            print(np.unique(labeled))
            if net==1:
                labeled_new, count = remove_small_islands(labeled, 5000)
            else:
                labeled_new, count = remove_small_islands(labeled, 500)
            print(np.unique(labeled_new))
            mask = labeled_new == 0
            print(new.max(), labeled_new.max())
            labeled_new = labeled_new + new.max()
            labeled_new[mask] = 0
            print(np.unique(labeled_new))
            new = new + labeled_new
            print(new.max(), labeled_new.max())
            ncs[i-1] = ncomponents - count
    else:
        new = np.zeros((s_final.shape))
        structure = np.ones((3, 3, 3), dtype=np.int)
        for i in range(1,int(s_final.max()+1)):
            strr = s_final == i
            labeled, _ = label(strr, structure)
            print(np.unique(labeled))
            labeled_new, _ = remove_small_islands(labeled, 1000)
            print(np.unique(labeled))
            new[labeled_new>0] = i
        ncs = 0


    np.save(data_results + '/ncs.npy',ncs)
    save_nii(np.asarray(new, dtype=np.int8), affine, newpath)
    print(name, 'CORRECTLY SAVED IN /PRED')

    '''
    for pp in range(3):
        pred = pred3d[pp]
        s_image__ = np.rollaxis(pred, 2, 0)
        s_image = np.rollaxis(s_image__, 2, 1)
        s = from_image_to_original(s_image, rrshape[2], rrshape[1], rrshape[0])
        img = nib.Nifti1Image(s, affine)
        nib.save(img, data_results + '/pred_' + str(pp) + '.nii.gz')
    '''




