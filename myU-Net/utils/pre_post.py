import os
import numpy as np
import shutil
from utils.pre_processing import load_nii, save_nii, resample_nii, seg_label_adults, rescaled
from utils.patches import adjust_dim, prepare_patches, from_patches_to_image, from_image_to_original_nii, create_patches, create_patches2
from utils.cropping import crop_to_bbox, crop_to_nonzero, adjust_to_bbox
from utils.losses import dice_post


#Working folders

def preprocessing(orig_path,img_path, label_path, mode):
    print('starting pre processing version 3.0 for ', img_path)
    #1.0 DATASET
    #
    #1.1 Load dataset and normalization
    if mode==2:
        dpatch_size = 1
        hpatch_size = 512
        wpatch_size = 512

        doverlap_stepsize = int(round(dpatch_size/2))
        if doverlap_stepsize == 0:
            doverlap_stepsize = 1
        hoverlap_stepsize = int(round(hpatch_size/2))
        woverlap_stepsize = int(round(wpatch_size/2))
    elif mode==3:
        dpatch_size = 96
        hpatch_size = 160
        wpatch_size = 160

        doverlap_stepsize = int(round(dpatch_size / 2))
        hoverlap_stepsize = int(round(hpatch_size / 2))
        woverlap_stepsize = int(round(wpatch_size / 2))


    res = [0.45703101, 0.45703101, 0.8999939]
    print('new resolution: ', res)

    if os.path.exists(orig_path + '/Slices'):
        shutil.rmtree(orig_path + '/Slices')
        os.mkdir(orig_path + '/Slices')
    else:
        os.mkdir(orig_path + '/Slices')
    newpath = orig_path + '/Slices'

    img, affine, hdr = load_nii(img_path)
    print(img.shape)
    print(hdr.get_zooms())
    img_crop, bbox = crop_to_nonzero(img, -300)
    img_r = resample_nii(img_crop, hdr, res, mode = 'image')
    print('after resampling: ', img_r.shape)
    img_r = rescaled(img_r)
    img_rr = np.rollaxis(img_r, 2, 0)
    img_rrr = np.rollaxis(img_rr, 2, 1)
    x_ = adjust_dim(img_rrr, dpatch_size, hpatch_size, wpatch_size)
    patch_x_ = prepare_patches(x_, dpatch_size, hpatch_size, wpatch_size, doverlap_stepsize,
                                           hoverlap_stepsize, woverlap_stepsize)
    x_images = create_patches(x_, patch_x_, dpatch_size, hpatch_size, wpatch_size)
    print('patches created')
    x_images = x_images.reshape(len(patch_x_), 1, dpatch_size, hpatch_size, wpatch_size)

    utils_for_post = [affine, hdr, patch_x_, x_.shape, img_rrr.shape, img_crop.shape, bbox, img.shape]

    print('new size: ', x_images.shape)
    seg, segaffine, seghdr = load_nii(label_path)
    seg_crop = crop_to_bbox(seg, bbox)
    labels = seg_label_adults(seg_crop)
    y_K = resample_nii(labels[:, :, :, 0], seghdr, res, mode="target")
    y_T = resample_nii(labels[:, :, :, 1], seghdr, res, mode="target")
    y_KK = np.rollaxis(y_K, 2, 0)
    y_KKK = np.rollaxis(y_KK, 2, 1)
    y_TT = np.rollaxis(y_T, 2, 0)
    y_TTT = np.rollaxis(y_TT, 2, 1)
    y_K_ = adjust_dim(y_KKK, dpatch_size, hpatch_size, wpatch_size)
    y_T_ = adjust_dim(y_TTT, dpatch_size, hpatch_size, wpatch_size)
    # print(y_K_.shape)
    # patch_y_ = prepare_patches(y_K_, dpatch_size, hpatch_size, wpatch_size, doverlap_stepsize,
    #                                hoverlap_stepsize, woverlap_stepsize)
    y_ = np.zeros([3, y_K_.shape[0], y_K_.shape[1], y_K_.shape[2]], dtype='float32')
    y_[0, :, :, :] = 1 - (y_K_ + y_T_)
    y_[1, :, :, :] = y_K_
    y_[2, :, :, :] = y_T_
    # print(len(patch_y_))
    y_images = create_patches2(y_, patch_x_, dpatch_size, hpatch_size, wpatch_size)
    # print(y_images.shape)
    y_images = y_images.reshape(len(patch_x_), 3, dpatch_size, hpatch_size, wpatch_size)
    for j in range(len(patch_x_)):
        if mode==2:
            xx = np.asarray(x_images[j], dtype=np.float32).squeeze(1)
            yy = np.asarray(y_images[j], dtype=np.float32).squeeze(1)
            np.savez(newpath + '/' + label_path[-15:-7] + '_' + str(j+1000) + '.npz', x=xx, y = yy)
        elif mode==3:
            xx = np.asarray(x_images[j], dtype=np.float32)
            yy = np.asarray(y_images[j], dtype=np.float32)
            np.savez(newpath + '/' + label_path[-15:-7] + '_' + str(j+100) + '.npz', x=xx, y = yy)
    del img_r
    del img_rr
    del y_K
    del y_KK
    del y_T
    del y_TT
    del x_images
    del y_images
    del y_

    return newpath, len(patch_x_), utils_for_post

def postprocessing(pred3d, label_name, data_results, name, utils_for_post, mode):

    affine, hdr, patch_ids, x_shape, rrshape, cropshape, bbox, shape = utils_for_post
    if mode==2:
        dpatch_size = 1
        hpatch_size = 512
        wpatch_size = 512
    elif mode==3:
        dpatch_size = 96
        hpatch_size = 160
        wpatch_size = 160


    res = [0.45703101, 0.45703101, 0.8999939]
    if not os.path.exists(data_results + '/Pred'):
        os.mkdir(data_results + '/Pred')
    newpath = data_results + '/Pred/pred_' + name


    print('starting postprocessing for prediction...')
    kidney_patches = pred3d[:,1,:,:,:]
    tumor_patches = pred3d[:,2,:,:,:]
    kidney_image_ = from_patches_to_image(kidney_patches, x_shape, patch_ids, dpatch_size, hpatch_size, wpatch_size)
    kidney_image__ = np.rollaxis(kidney_image_, 2, 0)
    kidney_image = np.rollaxis(kidney_image__, 2, 1)
    del kidney_image_
    del kidney_image__
    del kidney_patches
    kidney = from_image_to_original_nii(hdr, kidney_image, rrshape[2], rrshape[1], rrshape[0], res)
    del kidney_image
    kidney_ = adjust_dim(kidney, cropshape[0], cropshape[1], cropshape[2])
    del kidney
    kidney_final = adjust_to_bbox(kidney_, bbox, shape[0], shape[1], shape[2])
    del kidney_
    kidney_final[kidney_final <= 0.5] = 0.0
    kidney_final[kidney_final > 0.5] = 1.0
    tumor_image_ = from_patches_to_image(tumor_patches, x_shape, patch_ids, dpatch_size, hpatch_size, wpatch_size)
    tumor_image__ = np.rollaxis(tumor_image_, 2, 0)
    tumor_image = np.rollaxis(tumor_image__, 2, 1)
    del tumor_image_
    del tumor_image__
    del tumor_patches
    tumor = from_image_to_original_nii(hdr, tumor_image, rrshape[2], rrshape[1], rrshape[0], res)
    del tumor_image
    tumor_ = adjust_dim(tumor, cropshape[0], cropshape[1], cropshape[2])
    del tumor
    tumor_final = adjust_to_bbox(tumor_, bbox, shape[0], shape[1], shape[2])
    del tumor_
    tumor_final[tumor_final <= 0.5] = 0.0
    tumor_final[tumor_final > 0.5] = 1.0
    labe, _, _ = load_nii(label_name)
    labels = seg_label_adults(labe)
    dicekidney = dice_post(labels[:, :, :, 0], kidney_final)
    dicetumor = dice_post(labels[:, :, :, 1], tumor_final)
    print(name, '   kidney: ', round(dicekidney, 3), '   tumor: ', round(dicetumor, 3))
    tumor_final[tumor_final == 1.0] = 2.0
    predicted_image = kidney_final + tumor_final
    predicted_image[predicted_image == 3.0] = 1.0
    print(predicted_image.shape)
    save_nii(np.asarray(predicted_image, dtype=np.int8), affine, newpath)
    print(name, 'CORRECTLY SAVED IN /PRED')
    del kidney_final
    del tumor_final
    del predicted_image

    return dicekidney, dicetumor
