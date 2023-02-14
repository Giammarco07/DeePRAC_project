import os
import numpy as np
import shutil
from utils.pre_processing import load_nii, save_nii, resample_nii, seg_label, rescaled
from utils.patches import adjust_dim, prepare_patches, from_image_to_original_nii, creation_gauss_filter
from utils.cropping import crop_to_bbox, crop_to_nonzero, adjust_to_bbox
from utils.losses import dice_post
from scipy.ndimage import gaussian_filter
from utils.figures import ensemble_np
from medpy.metric.binary import hd

def preprocessing_a(img_path, label_path, channel_dim, mode,res,preprocessing, do_seg=False):
    print('starting pre processing version 3.0 for ', img_path)
    #1.0 DATASET
    #
    #1.1 Load dataset and normalization
    if mode==2:
        dpatch_size = 1
        hpatch_size = 512
        wpatch_size = 512

        doverlap_stepsize = 1
        hoverlap_stepsize = 512
        woverlap_stepsize = 512
    elif mode==3:
        dpatch_size = 128
        hpatch_size = 128
        wpatch_size = 128

        doverlap_stepsize = int(round(dpatch_size / 2))
        hoverlap_stepsize = int(round(hpatch_size / 2))
        woverlap_stepsize = int(round(wpatch_size / 2))

    img, affine, hdr = load_nii(img_path)
    print(img.shape)
    print(hdr.get_zooms())
    img_crop, bbox = crop_to_nonzero(img, -300)
    print(img_crop.shape)
    if mode==2:
        img_r = resample_nii(img_crop, hdr, res, mode='image2D')
        print('depth dimension will not be modified')
    else:
        img_r = resample_nii(img_crop, hdr, res, mode = 'image')

    print('new resolution: ', res)
    print('after resampling: ', img_r.shape)

    #img_r = gaussian_filter(img_r, sigma=1)

    img_rrr = rescaled(img_r, preprocessing)

    x_ = adjust_dim(img_rrr, dpatch_size, hpatch_size, wpatch_size)
    print('final dim: ', x_.shape)
    patch_x_ = prepare_patches(x_, dpatch_size, hpatch_size, wpatch_size, doverlap_stepsize,
                               hoverlap_stepsize, woverlap_stepsize)

    utils_for_post = [affine, hdr, patch_x_, x_.shape, img_rrr.shape, img_crop.shape, bbox, img.shape]

    if do_seg:
        print('doing the same for reference segmentation...')
        seg, segaffine, seghdr = load_nii(label_path)
        seg_crop = crop_to_bbox(seg, bbox)
        labels = seg_label(seg_crop, channel_dim)
        if mode == 2:
            y_KKK = resample_nii(labels[:, :, :, 0], seghdr, res, mode="target2D")
        else:
            y_KKK = resample_nii(labels[:, :, :, 0], seghdr, res, mode="target")
        y_K_ = adjust_dim(y_KKK, dpatch_size, hpatch_size, wpatch_size)
        if channel_dim > 2:
            if mode == 2:
                y_TTT = resample_nii(labels[:, :, :, 1], seghdr, res, mode="target2D")
            else:
                y_TTT = resample_nii(labels[:, :, :, 1], seghdr, res, mode="target")
            y_T_ = adjust_dim(y_TTT, dpatch_size, hpatch_size, wpatch_size)
        if channel_dim > 3:
            if mode == 2:
                y_CCC = resample_nii(labels[:, :, :, 2], seghdr, res, mode="target2D")
            else:
                y_CCC = resample_nii(labels[:, :, :, 2], seghdr, res, mode="target")
            y_C_ = adjust_dim(y_CCC, dpatch_size, hpatch_size, wpatch_size)
        y_ = np.zeros([channel_dim, y_K_.shape[0], y_K_.shape[1], y_K_.shape[2]], dtype='float16')
        y_[0, :, :, :] = 1 - (y_K_)
        y_[1, :, :, :] = y_K_
        if channel_dim > 2:
            y_[2, :, :, :] = y_T_
            y_[0, :, :, :] -= y_T_
        if channel_dim > 3:
            y_[3, :, :, :] = y_C_
            y_[0, :, :, :] -= y_C_

    #del img_r

    if do_seg:
        return len(patch_x_), utils_for_post, x_, y_
    else:
        return len(patch_x_), utils_for_post, x_

def postprocessing_a(pred3d, label_name, data_results, name, utils_for_post,  channel_dim, mode,res):
    print('starting postprocessing for prediction...')

    affine, hdr, patch_ids, x_shape, rrshape, cropshape, bbox, shape = utils_for_post
    #affine, hdr, patch_ids, x_shape, rrshape, shape = utils_for_post

    if mode == 2:
        dpatch_size = 1
        hpatch_size = 512
        wpatch_size = 512
    elif mode == 3:
        dpatch_size = 128
        hpatch_size = 128
        wpatch_size = 128

    if not os.path.exists(data_results + '/Pred'):
        os.mkdir(data_results + '/Pred')
    newpath = data_results + '/Pred/pred_' + name

    labe, _, _ = load_nii(label_name)
    labels = seg_label(labe, channel_dim)


    gauss = creation_gauss_filter(x_shape, patch_ids, dpatch_size, hpatch_size, wpatch_size)
    gauss_n = np.repeat(gauss[np.newaxis, :, :, :], channel_dim, axis=0)
    pred3d = pred3d / gauss_n


    del gauss
    del gauss_n

    dicetot = []
    msetot = []
    hdtot = []

    s_image = np.argmax(pred3d, axis=0)
    #s_final = np.argmax(pred3d, axis=0)


    s = from_image_to_original_nii(hdr, s_image, rrshape[0], rrshape[1], rrshape[2], res,mode)
    #s_final = from_image_to_original_nii(hdr, s_image, rrshape[0], rrshape[1], rrshape[2], res, mode)
    del s_image


    s_ = adjust_dim(s, cropshape[0], cropshape[1], cropshape[2])
    del s
    s_final = adjust_to_bbox(s_, bbox, shape[0], shape[1], shape[2])
    del s_
    s_final = np.rint(s_final)

    print(name)
    print(s_final.shape)
    s_finals = seg_label(s_final, channel_dim)
    for i in range(1, channel_dim):
        dice = dice_post(labels[:, :, :, (i - 1)], s_finals[:, :, :, (i - 1)])
        dicetot.append(dice)

        print('Structure ', str(i), ' : ', round(dice, 3))
        
        mserror = (np.square(labels[:, :, :, (i - 1)] - s_finals[:, :, :, (i - 1)])).mean(axis=None)
        msetot.append(mserror)
        # A non-negative floating point value (the best value is 0.0).
        if (labels[:, :, :, (i - 1)].sum())!=0 and (s_finals[:, :, :, (i - 1)].sum())!=0:
            hdistance = hd(s_finals[:, :, :, (i - 1)], labels[:, :, :, (i - 1)])
        elif (labels[:, :, :, (i - 1)].sum())!=0 and (s_finals[:, :, :, (i - 1)].sum())==0:
            hdistance = 1000.0
        else:
            hdistance = 0.0
        hdtot.append(hdistance)
        #The symmetric Hausdorff Distance between the object(s) in `result` and the object(s) in `reference`.
        #The distance unit is the same as for the spacing of elements along each dimension, which is usually given in mm.
        
        

    save_nii(np.asarray(s_final, dtype=np.int8), affine, newpath)
    print(name, 'CORRECTLY SAVED IN /PRED')

    return dicetot, msetot, hdtot
