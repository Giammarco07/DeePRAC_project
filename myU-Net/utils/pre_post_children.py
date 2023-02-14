import os
import numpy as np
import shutil
from utils.pre_processing import load_nii, save_nii, resample_nii, seg_label, rescaled, seg_label_children
from utils.patches import adjust_dim, prepare_patches, from_image_to_original_nii, creation_gauss_filter, creation_gauss_filter_skel
from utils.cropping import crop_to_bbox, crop_to_nonzero, adjust_to_bbox
from utils.losses import dice_post
from medpy.metric.binary import hd95 as hd
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from utils.patches import gaussian_map

def preprocessing_c(img_path, label_path, patch_size, channel_dim, in_c, mode, res, preprocessing, do_seg=True, rsz=False, input=None):
    print('starting pre processing version 3.0 for ', img_path)
    #1.0 DATASET
    #
    #1.1 Load dataset and normalization
    if mode==2:
        dpatch_size = 1
        hpatch_size = patch_size[0]
        wpatch_size = patch_size[1]

        doverlap_stepsize = int(round(dpatch_size/2))
        if doverlap_stepsize == 0:
            doverlap_stepsize = 1
        hoverlap_stepsize = int(round(hpatch_size/2))
        woverlap_stepsize = int(round(wpatch_size/2))

    elif mode==3:
        dpatch_size = patch_size[0]
        hpatch_size = patch_size[1]
        wpatch_size = patch_size[2]
        if rsz:
            dpatch_size = 512
            hpatch_size = 512
            wpatch_size = 512


        doverlap_stepsize = int(round(dpatch_size / 2)) 
        hoverlap_stepsize = int(round(hpatch_size / 2)) 
        woverlap_stepsize = int(round(wpatch_size / 2)) 

    img, affine, hdr = load_nii(img_path)
    print(img.shape)
    #print(hdr.get_zooms())

    seg, segaffine, seghdr = load_nii(label_path)
    
    
    
    if mode==3 and rsz==False:
        _, bbox = crop_to_nonzero(seg, 0)
        img_rrr = crop_to_bbox(img, bbox)
        if img_rrr.shape[2] % dpatch_size != 0:
                d = dpatch_size - (img_rrr.shape[2] % dpatch_size)
        else:
                d = 0
        if img_rrr.shape[1] % hpatch_size != 0:
                h = hpatch_size - (img_rrr.shape[1] % hpatch_size)
        else:
                h = 0
        if img_rrr.shape[0] % wpatch_size != 0:
                w = wpatch_size - (img_rrr.shape[0] % wpatch_size)
        else:
                w = 0   

        print(bbox)
        bbox_adj = [[bbox[0][0]-w//2,bbox[0][1]+w//2],[bbox[1][0]-h//2,bbox[1][1]+h//2],[bbox[2][0]-d//2,bbox[2][1]+d//2]]
        print(bbox_adj)
        bbox_new = [[np.max((bbox_adj[0][0],0)),np.min((bbox_adj[0][1],img.shape[0]-1))],[np.max((bbox_adj[1][0],0)),np.min((bbox_adj[1][1],img.shape[1]-1))],[np.max((bbox_adj[2][0],0)),np.min((bbox_adj[2][1],img.shape[2]-1))]]
        print(bbox_new)
        img_crop = crop_to_bbox(img, bbox_new)
        seg_crop = crop_to_bbox(seg, bbox_new)
        print(seg_crop.shape) 
    else:
        img_crop, bbox_new = crop_to_nonzero(img, -300)
        seg_crop = crop_to_bbox(seg, bbox_new)
        print(seg_crop.shape)
    

    
    #print(seg.shape)
    #print(np.unique(seg))
    
    #seg = seg_label_children(seg)
    
    #if patch_size[0]==32:
    #seg_crop, bbox = crop_to_nonzero(seg, 0)
    #print(seg_crop.shape)
    #img_crop = crop_to_bbox(img, bbox)
    #else:
    #    img_crop, bbox = crop_to_nonzero(img, -300)
        
    print('after crop:' , img_crop.shape)
    if rsz==True:
        img_r = img_crop
    elif mode==2 and rsz==False:
        img_r = resample_nii(img_crop, hdr, res, mode='image2D')
        #print('depth dimension will not be modified')
    else:
        img_r = resample_nii(img_crop, hdr, res, mode = 'image')
    #print('new resolution: ', res)
    print('after resampling: ', img_r.shape)
    #img_r = gaussian_filter(img_r, sigma=0.5)
    if in_c == 2:
        img_r = rescaled(img_r, preprocessing[0])
    else:
        img_r = rescaled(img_r, preprocessing)
    if input == 'adults':
        img_rrr = img_r[:,::-1,:]
    else:
        img_rr = np.rollaxis(img_r, 2, 0)
        img_rrr = np.rollaxis(img_rr, 2, 1)
    print('adjusting dimension: ', img_rrr.shape)



    if img_rrr.shape[0] % dpatch_size != 0:
        d = dpatch_size - (img_rrr.shape[0] % dpatch_size)
    else:
        d = 0
    if img_rrr.shape[1] % hpatch_size != 0:
        h = hpatch_size - (img_rrr.shape[1] % hpatch_size)
    else:
        h = 0
    if img_rrr.shape[2] % wpatch_size != 0:
        w = wpatch_size - (img_rrr.shape[2] % wpatch_size)
    else:
        w = 0
        
    x_ = adjust_dim(img_rrr, img_rrr.shape[0] + d, img_rrr.shape[1] + h, img_rrr.shape[2] + w)
    print('final dim: ',x_.shape)
    patch_x_ = prepare_patches(x_, dpatch_size, hpatch_size, wpatch_size, doverlap_stepsize,
                                           hoverlap_stepsize, woverlap_stepsize)

    if in_c ==2:
        #img, _, _ = load_nii(img_path[:-8] + '1.nii.gz')
        #print(img.shape)
        #print(hdr.get_zooms())
        #img_crop, bbox = crop_to_nonzero(img, -300)
        #img_crop = crop_to_bbox(img, bbox)
        #print('after crop:', img_crop.shape)
        #img_r = resample_nii(img_crop, hdr, res, mode='image')
        #print('new resolution: ', res)
        #print('after resampling: ', img_r.shape)
        #img_r = rescaled(img_r, preprocessing[1])
        #img_rr = np.rollaxis(img_r, 2, 0)
        #img_rrr = np.rollaxis(img_rr, 2, 1)
        #print('adjusting dimension: ', img_rrr.shape)
        #x_1 = adjust_dim(img_rrr, img_rrr.shape[0] + d, img_rrr.shape[1] + h,
        #                img_rrr.shape[2] + w)
        img_c = np.copy(x_)
        maximum, minimum = img_c.max(),img_c.min()
        img_c[img_c>=0] +=  0.2*img_c[img_c>=0]
        img_c[img_c<0] -= 0.2*img_c[img_c<0]
        img_c = np.clip(img_c,minimum,maximum)
        x_1 = frangi(rescale(img_c), scale_range=(3,8), scale_step=5/4, alpha=0.5, beta=0.5, gamma=15, black_ridges=False)
        x_1[x_1>0]=1

        x_= np.asarray([x_, x_1], dtype=np.float32)
    else:
        x_ = x_[np.newaxis, ...]


    utils_for_post = [affine, hdr, patch_x_, x_.shape, img_rrr.shape, img_crop.shape, bbox_new, img.shape]

    if do_seg:
        print('doing the same for reference segmentation...')
        seg, segaffine, seghdr = load_nii(label_path)
        seg_crop = crop_to_bbox(seg, bbox_new)
        labels = seg_label(seg_crop, channel_dim)
        if rsz==True:
            y_K = labels[:, :, :, 0]
        elif mode == 2 and rsz==False:
            y_K = resample_nii(labels[:, :, :, 0], seghdr, res, mode="target2D")
        else:
            y_K = resample_nii(labels[:, :, :, 0], seghdr, res, mode="target")
        y_KK = np.rollaxis(y_K, 2, 0)
        y_KKK = np.rollaxis(y_KK, 2, 1)
        y_K_ = adjust_dim(y_KKK, img_rrr.shape[0] + d, img_rrr.shape[1] + h, img_rrr.shape[2] + w)
        if channel_dim>2:
            if rsz==True:
                y_T = labels[:, :, :, 1]
            elif mode == 2 and rsz==False:
                y_T = resample_nii(labels[:, :, :, 1], seghdr, res, mode="target2D")
            else:
                y_T = resample_nii(labels[:, :, :, 1], seghdr, res, mode="target")
            y_TT = np.rollaxis(y_T, 2, 0)
            y_TTT = np.rollaxis(y_TT, 2, 1)
            y_T_ = adjust_dim(y_TTT, img_rrr.shape[0] + d, img_rrr.shape[1] + h, img_rrr.shape[2] + w)
        if channel_dim>3:
            if mode == 2:
                y_C = resample_nii(labels[:, :, :, 2], seghdr, res, mode="target2D")
            else:
                y_C = resample_nii(labels[:, :, :, 2], seghdr, res, mode="target")
            y_CC = np.rollaxis(y_C, 2, 0)
            y_CCC = np.rollaxis(y_CC, 2, 1)
            y_C_ = adjust_dim(y_CCC, img_rrr.shape[0] + d, img_rrr.shape[1] + h, img_rrr.shape[2] + w)
        y_ = np.zeros([channel_dim, y_K_.shape[0], y_K_.shape[1], y_K_.shape[2]], dtype='float32')
        y_[0, :, :, :] = 1 - (y_K_)
        y_[1, :, :, :] = y_K_
        del y_K
        del y_KK
        if channel_dim>2:
            y_[2, :, :, :] = y_T_
            y_[0, :, :, :] -= y_T_
            del y_T
            del y_TT
        if channel_dim>3:
            y_[3, :, :, :] = y_C_
            y_[0, :, :, :] -= y_C_
            del y_C
            del y_CC

    del img_r
    del img_rr

    if do_seg:
        return len(patch_x_), utils_for_post, x_, y_
    else:
        return len(patch_x_), utils_for_post, x_


def preprocessing_c_p(img_path, label_path, channel_dim, mode, res, preprocessing):
    print('starting pre processing version 3.0 for ', img_path)
    # 1.0 DATASET
    #
    # 1.1 Load dataset and normalization
    if mode == 2:
        dpatch_size = 1
        hpatch_size = 512
        wpatch_size = 512

        doverlap_stepsize = int(round(dpatch_size / 2))
        if doverlap_stepsize == 0:
            doverlap_stepsize = 1
        hoverlap_stepsize = int(round(hpatch_size / 2))
        woverlap_stepsize = int(round(wpatch_size / 2))

    elif mode == 3:
        dpatch_size = 32
        hpatch_size = 64
        wpatch_size = 64

        doverlap_stepsize = int(round(dpatch_size / 2))
        hoverlap_stepsize = int(round(hpatch_size / 2))
        woverlap_stepsize = int(round(wpatch_size / 2))

    img, affine, hdr = load_nii(img_path)
    print(img.shape)
    print(hdr.get_zooms())

    seg, segaffine, seghdr = load_nii(label_path)
    print(seg.shape)
    print(np.unique(seg))

    # seg = seg_label_children(seg)

    #seg_crop, bbox = crop_to_nonzero(seg, 0)
    #print(seg_crop.shape)
    #img_crop = crop_to_bbox(img, bbox)

    img_crop, bbox = crop_to_nonzero(img, -300)
    print('after crop:', img_crop.shape)
    if mode == 2:
        img_r = resample_nii(img_crop, hdr, res, mode='image2D')
        print('depth dimension will not be modified')
    else:
        img_r = resample_nii(img_crop, hdr, res, mode='image')
    print('new resolution: ', res)
    print('after resampling: ', img_r.shape)

    img_r = rescaled(img_r, preprocessing)
    img_rr = np.rollaxis(img_r, 2, 0)
    img_rrr = np.rollaxis(img_rr, 2, 1)
    x_ = adjust_dim(img_rrr, dpatch_size, hpatch_size, wpatch_size)
    print('final dim: ', x_.shape)
    patch_x_ = prepare_patches(x_, dpatch_size, hpatch_size, wpatch_size, doverlap_stepsize,
                               hoverlap_stepsize, woverlap_stepsize)

    utils_for_post = [affine, hdr, patch_x_, x_.shape, img_rrr.shape, img_crop.shape, bbox, img.shape]


    print('doing the same for reference segmentation...')
    seg, segaffine, seghdr = load_nii(label_path)
    seg_crop = crop_to_bbox(seg, bbox)
    labels = seg_label(seg_crop, channel_dim)
    if mode == 2:
        y_K = resample_nii(labels[:, :, :, 0], seghdr, res, mode="target2D")
    else:
        y_K = resample_nii(labels[:, :, :, 0], seghdr, res, mode="target")
    y_KK = np.rollaxis(y_K, 2, 0)
    y_KKK = np.rollaxis(y_KK, 2, 1)
    y_K_ = adjust_dim(y_KKK, dpatch_size, hpatch_size, wpatch_size)
    if channel_dim > 2:
        if mode == 2:
            y_T = resample_nii(labels[:, :, :, 1], seghdr, res, mode="target2D")
        else:
            y_T = resample_nii(labels[:, :, :, 1], seghdr, res, mode="target")
        y_TT = np.rollaxis(y_T, 2, 0)
        y_TTT = np.rollaxis(y_TT, 2, 1)
        y_T_ = adjust_dim(y_TTT, dpatch_size, hpatch_size, wpatch_size)
    if channel_dim > 3:
        if mode == 2:
            y_C = resample_nii(labels[:, :, :, 2], seghdr, res, mode="target2D")
        else:
            y_C = resample_nii(labels[:, :, :, 2], seghdr, res, mode="target")
        y_CC = np.rollaxis(y_C, 2, 0)
        y_CCC = np.rollaxis(y_CC, 2, 1)
        y_C_ = adjust_dim(y_CCC, dpatch_size, hpatch_size, wpatch_size)
    y_ = np.zeros([channel_dim, y_K_.shape[0], y_K_.shape[1], y_K_.shape[2]], dtype='float32')
    y_[0, :, :, :] = 1 - (y_K_)
    y_[1, :, :, :] = y_K_
    del y_K
    del y_KK
    if channel_dim > 2:
        y_[2, :, :, :] = y_T_
        y_[0, :, :, :] -= y_T_
        del y_T
        del y_TT
    if channel_dim > 3:
        y_[3, :, :, :] = y_C_
        y_[0, :, :, :] -= y_C_
        del y_C
        del y_CC

    del img_r
    del img_rr

    return len(patch_x_), utils_for_post, x_, y_


def preprocessing_skel(img_path, label_path, channel_dim, mode, res, preprocessing, rsz=False, input=None):
    import sknw
    from skimage.morphology import skeletonize
    print('starting pre processing version 3.0 for ', img_path)
    #1.0 DATASET
    #
    #1.1 Load dataset and normalization
   

    img, affine, hdr = load_nii(img_path)
    print(img.shape)
    print(hdr.get_zooms())

    seg, segaffine, seghdr = load_nii(label_path)
    
    img_crop, bbox = crop_to_nonzero(img, -300)
    seg_crop = crop_to_bbox(seg, bbox)

    #
    print('after crop:' , img_crop.shape)
    if mode==2:
        img_r = resample_nii(img_crop, hdr, res, mode='image2D')
        print('depth dimension will not be modified')
    else:
        img_r = resample_nii(img_crop, hdr, res, mode = 'image')
    print('new resolution: ', res)
    print('after resampling: ', img_r.shape)
    #img_r = gaussian_filter(img_r, sigma=0.5)
    img_r = rescaled(img_r, preprocessing)
    img_rr = np.rollaxis(img_r, 2, 0)
    x_ = np.rollaxis(img_rr, 2, 1)

    print('final dim: ',x_.shape)
    
    labels = seg_label(seg_crop, channel_dim)
    if mode == 2:
        y_K = resample_nii(labels[:, :, :, 0], seghdr, res, mode="target2D")
    else:
        y_K = resample_nii(labels[:, :, :, 0], seghdr, res, mode="target")
    y_KK = np.rollaxis(y_K, 2, 0)
    y_K_ = np.rollaxis(y_KK, 2, 1)
    
    if channel_dim>2:
        if mode == 2:
            y_T = resample_nii(labels[:, :, :, 1], seghdr, res, mode="target2D")
        else:
            y_T = resample_nii(labels[:, :, :, 1], seghdr, res, mode="target")
        y_TT = np.rollaxis(y_T, 2, 0)
        y_T_ = np.rollaxis(y_TT, 2, 1)
        
    if channel_dim>3:
        if mode == 2:
            y_C = resample_nii(labels[:, :, :, 2], seghdr, res, mode="target2D")
        else:
            y_C = resample_nii(labels[:, :, :, 2], seghdr, res, mode="target")
        y_CC = np.rollaxis(y_C, 2, 0)
        y_C_ = np.rollaxis(y_CC, 2, 1)
        
    y_ = np.zeros([channel_dim, y_K_.shape[0], y_K_.shape[1], y_K_.shape[2]], dtype='float32')
    y_[0, :, :, :] = 1 - (y_K_)
    y_[1, :, :, :] = y_K_
    del y_K
    del y_KK
    if channel_dim>2:
        y_[2, :, :, :] = y_T_
        y_[0, :, :, :] -= y_T_
        del y_T
        del y_TT
    if channel_dim>3:
        y_[3, :, :, :] = y_C_
        y_[0, :, :, :] -= y_C_
        del y_C
        del y_CC
        
    seg_rr =  np.argmax(y_, axis=0)

    print(np.unique(seg_rr))
    n_struct = seg_rr.max().astype(int)
    patch_ids = []
    for struct in range(1,n_struct+1):
        dat = seg_rr == struct
        skel = skeletonize(dat)
        skel = skel / skel.max()
        graph = sknw.build_sknw(skel)
        for (s, e) in graph.edges():
            ps = graph[s][e]['pts']
            nps = ps.shape[0]
            patch_ids.append(tuple(ps[0]))
            if nps>15:
                ind = 15
                while ind<nps:
                    patch_ids.append(tuple(ps[ind]))
                    ind += 16
            patch_ids.append(tuple(ps[-1]))

    utils_for_post = [affine, hdr, patch_ids, x_.shape, x_.shape, img_crop.shape, bbox, img.shape]



    del img_r
    del img_rr

    return len(patch_ids), utils_for_post, x_, y_

def preprocessing_c25(img_path, label_path, channel_dim, res, preprocessing):
    print('starting pre processing version 3.0 for ', img_path)

    img, affine, hdr = load_nii(img_path)
    print(img.shape)
    print(hdr.get_zooms())
    img_r = resample_nii(img, hdr, res, mode = 'image')

    print('new resolution: ', res)
    print('after resampling: ', img_r.shape)
    img_r = rescaled(img_r, preprocessing)
    img_rr = np.rollaxis(img_r, 2, 0)
    img_rrr = np.rollaxis(img_rr, 2, 1)
    print('final shape: ',img_rrr.shape)

    seg, segaffine, seghdr = load_nii(label_path)
    labels = seg_label(seg, channel_dim)
    y_K = resample_nii(labels[:, :, :, 0], seghdr, res, mode="target")
    y_KK = np.rollaxis(y_K, 2, 0)
    y_KKK = np.rollaxis(y_KK, 2, 1)
    if channel_dim > 2:
        y_T = resample_nii(labels[:, :, :, 1], seghdr, res, mode="target")
        y_TT = np.rollaxis(y_T, 2, 0)
        y_TTT = np.rollaxis(y_TT, 2, 1)
    if channel_dim > 3:
        y_C = resample_nii(labels[:, :, :, 2], seghdr, res, mode="target")
        y_CC = np.rollaxis(y_C, 2, 0)
        y_CCC = np.rollaxis(y_CC, 2, 1)
    y_ = np.zeros([channel_dim, y_KKK.shape[0], y_KKK.shape[1], y_KKK.shape[2]], dtype='float32')
    y_[0, :, :, :] = 1 - (y_KKK)
    y_[1, :, :, :] = y_KKK
    del y_K
    del y_KK
    if channel_dim > 2:
        y_[2, :, :, :] = y_TTT
        y_[0, :, :, :] -= y_TTT
        del y_T
        del y_TT
    if channel_dim > 3:
        y_[3, :, :, :] = y_CCC
        y_[0, :, :, :] -= y_CCC
        del y_C
        del y_CC

    data = y_[0,:, :, :] < 1

    print('final seg shape: ', data.shape)


    utils_for_post = [affine, hdr, img.shape]

    del img_r
    del img_rr

    return img_rrr, data, y_, utils_for_post



def postprocessing_c(pred3d, label_name, patch_size, data_results, name, utils_for_post, channel_dim, mode, res, rsz=False, input=None):
    print('starting postprocessing for prediction...')

    affine, hdr, patch_ids, x_shape, rrshape, cropshape, bbox, shape = utils_for_post
    if mode==2:
        dpatch_size = 1
        hpatch_size = patch_size[0]
        wpatch_size = patch_size[1]
    elif mode==3:
        dpatch_size = patch_size[0]
        hpatch_size = patch_size[1]
        wpatch_size = patch_size[2]
        if rsz:
            dpatch_size = 512
            hpatch_size = 512
            wpatch_size = 512


    if not os.path.exists(data_results + '/Pred'):
        os.mkdir(data_results + '/Pred')
    newpath = data_results + '/Pred/pred_' + name

    labe, _, _ = load_nii(label_name)

    #np.save(newpath[:-7], np.asarray(pred3d, dtype=np.float16))

    gauss = creation_gauss_filter(x_shape, patch_ids, dpatch_size, hpatch_size, wpatch_size)
    gauss_n = np.repeat(gauss[np.newaxis, :, :, :], channel_dim, axis=0).astype(np.float32)
    pred3d = pred3d / gauss_n

    del gauss
    del gauss_n

    dicetot = []
    precisiontot = []
    recalltot = []
    hdtot = []

    pred = np.argmax(pred3d, axis=0)

    if input=='adults':
        pred = pred[:,::-1,:]
    s_image__ = np.rollaxis(pred, 2, 0)
    s_image = np.rollaxis(s_image__, 2, 1)

    del s_image__
    s = from_image_to_original_nii(hdr, s_image, rrshape[2], rrshape[1], rrshape[0], cropshape[0], cropshape[1], cropshape[2], res,mode)
    print(s.shape)
    del s_image
    #s_ = adjust_dim(s, cropshape[0], cropshape[1], cropshape[2])
    #del s
    s_final = adjust_to_bbox(s, bbox, shape[0], shape[1], shape[2])
    del s
    s_final = np.rint(s_final).astype(np.uint8)

    print(name)
    print(s_final.shape)
    labes, bbox = crop_to_nonzero(labe, 0)
    labels = seg_label(labes, channel_dim)
    s_finalss = crop_to_bbox(s_final, bbox)
    s_finals = seg_label(s_finalss, channel_dim)
    
    for i in range(1,channel_dim):
        dice = dice_post(labels[:, :, :, (i-1)], s_finals[:, :, :, (i-1)])
        dicetot.append(dice)

        precisionerror = (np.sum(labels[:, :, :, (i - 1)] * s_finals[:, :, :, (i - 1)])/np.sum(s_finals[:, :, :, (i - 1)]))
        precisiontot.append(precisionerror)
        recallerror = (np.sum(labels[:, :, :, (i - 1)] * s_finals[:, :, :, (i - 1)])/np.sum(labels[:, :, :, (i - 1)]))
        recalltot.append(recallerror)
        # A non-negative floating point value (the best value is 0.0).
        if (labels[:, :, :, (i - 1)].sum())!=0 and (s_finals[:, :, :, (i - 1)].sum())!=0:
            hdistance = hd(s_finals[:, :, :, (i - 1)], labels[:, :, :, (i - 1)],voxelspacing = hdr.get_zooms())
        elif (labels[:, :, :, (i - 1)].sum())!=0 and (s_finals[:, :, :, (i - 1)].sum())==0:
            hdistance = 1000.0
        else:
            hdistance = 0.0
        hdtot.append(hdistance)
        #The symmetric Hausdorff Distance between the object(s) in `result` and the object(s) in `reference`.
        #The distance unit is the same as for the spacing of elements along each dimension, which is usually given in mm.

        print('Structure ', str(i), ' : ', round(dice, 3))


    save_nii(np.asarray(s_final, dtype=np.int8), affine, newpath)
    print(name, 'CORRECTLY SAVED IN /PRED')


    return dicetot, precisiontot, recalltot, hdtot


def postprocessing_c_p(pred3d, data_results, name, utils_for_post, mode, res):
    print('starting postprocessing for prediction...')

    affine, hdr, patch_ids, x_shape, rrshape, cropshape, bbox, shape = utils_for_post


    if not os.path.exists(data_results + '/Pred'):
        os.mkdir(data_results + '/Pred')
    newpath = data_results + '/Pred/' + name

    pred3d[pred3d<0] = 0
    pred3d[pred3d>1] = 1

    s_image__ = np.rollaxis(pred3d, 2, 0)
    s_image = np.rollaxis(s_image__, 2, 1)
    del s_image__
    s = from_image_to_original_nii(hdr, s_image, rrshape[2], rrshape[1], rrshape[0], res, mode)
    del s_image
    s_ = adjust_dim(s, cropshape[0], cropshape[1], cropshape[2])
    del s
    s_final = adjust_to_bbox(s_, bbox, shape[0], shape[1], shape[2])
    del s_
    s_final = np.rint(s_final).astype(np.uint8)

    print(name)
    print(s_final.shape)

    save_nii(np.asarray(s_final, dtype=np.int8), affine, newpath)
    print(name, 'CORRECTLY SAVED IN /PRED')



def postprocessing_c_v(pred3d, label_name, data_results, name, utils_for_post, channel_dim, mode, res, v=None):
    print('starting postprocessing for prediction...')

    affine, hdr, patch_ids, x_shape, rrshape, cropshape, bbox, shape = utils_for_post
    if mode == 2:
        dpatch_size = 1
        hpatch_size = 512
        wpatch_size = 512
    elif mode == 3:
        dpatch_size = 96
        hpatch_size = 160
        wpatch_size = 160

    if not os.path.exists(data_results + '/Pred'):
        os.mkdir(data_results + '/Pred')
    newpath = data_results + '/Pred/pred_' + v + '_' + name

    gauss = creation_gauss_filter(x_shape, patch_ids, dpatch_size, hpatch_size, wpatch_size)
    gauss_n = gauss[np.newaxis, :, :, :].astype(np.float32)
    pred3d = pred3d / gauss_n

    del gauss
    del gauss_n

    print(pred3d.shape)
    pred = np.squeeze(pred3d, axis=0)

    s_image__ = np.rollaxis(pred, 2, 0)
    s_image = np.rollaxis(s_image__, 2, 1)

    del s_image__
    s = from_image_to_original_nii(hdr, s_image, rrshape[2], rrshape[1], rrshape[0], res, mode)
    del s_image
    s_ = adjust_dim(s, cropshape[0], cropshape[1], cropshape[2])
    del s
    s_final = adjust_to_bbox(s_, bbox, shape[0], shape[1], shape[2])
    del s_
    s_final = np.rint(s_final).astype(np.uint8)

    print(name)
    print(s_final.shape)

    save_nii(np.asarray(s_final, dtype=np.int8), affine, newpath)
    print(name, 'CORRECTLY SAVED IN /PRED')

def postprocessing_skel(pred3d, label_name, data_results, name, utils_for_post, channel_dim, mode, res, rsz=False, input=None):
    print('starting postprocessing for prediction...')

    affine, hdr, patch_ids, x_shape, rrshape, cropshape, bbox, shape = utils_for_post
    dpatch_size = 32
    hpatch_size = 64
    wpatch_size = 64


    if not os.path.exists(data_results + '/Pred'):
        os.mkdir(data_results + '/Pred')
    newpath = data_results + '/Pred/pred_' + name

    labe, _, _ = load_nii(label_name)
    
    #labe = seg_label_children(labe)
    
    labels = seg_label(labe, channel_dim)

    #np.save(newpath[:-7], np.asarray(pred3d, dtype=np.float16))

    gauss = creation_gauss_filter_skel(x_shape, patch_ids, dpatch_size, hpatch_size, wpatch_size)
    gauss_n = np.repeat(gauss[np.newaxis, :, :, :], channel_dim, axis=0).astype(np.float32)
    pred3d = pred3d / gauss_n

    del gauss
    del gauss_n

    dicetot = []
    msetot = []
    hdtot = []

    pred = np.argmax(pred3d, axis=0)

    if input=='adults':
        pred = pred[:,::-1,:]
    s_image__ = np.rollaxis(pred, 2, 0)
    s_image = np.rollaxis(s_image__, 2, 1)

    del s_image__
    s = from_image_to_original_nii(hdr, s_image, rrshape[2], rrshape[1], rrshape[0], res,mode)
    del s_image
    s_ = adjust_dim(s, cropshape[0], cropshape[1], cropshape[2])
    del s
    s_final = adjust_to_bbox(s_, bbox, shape[0], shape[1], shape[2])
    del s_
    s_final = np.rint(s_final).astype(np.uint8)

    print(name)
    print(s_final.shape)
    s_finals = seg_label(s_final, channel_dim)
    for i in range(1,channel_dim):
        dice = dice_post(labels[:, :, :, (i-1)], s_finals[:, :, :, (i-1)])
        dicetot.append(dice)

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

        print('Structure ', str(i), ' : ', round(dice, 3))


    save_nii(np.asarray(s_final, dtype=np.int8), affine, newpath)
    print(name, 'CORRECTLY SAVED IN /PRED')


    return dicetot, msetot, hdtot

def postprocessing_c25(pred3d, label_name, data_results, name, utils_for_post, channel_dim, res):
    print('starting postprocessing for prediction...')

    affine, hdr, imgshape = utils_for_post

    if not os.path.exists(data_results + '/Pred'):
        os.mkdir(data_results + '/Pred')
    newpath = data_results + '/Pred/pred_' + name

    labe, _, _ = load_nii(label_name)
    labels = seg_label(labe, channel_dim)

    dicetot = []
    msetot = []
    hdtot = []

    pred = np.argmax(pred3d, axis=0)

    s_image__ = np.rollaxis(pred, 2, 0)
    s_image = np.rollaxis(s_image__, 2, 1)

    del s_image__
    s = resize(s_image, imgshape, order=0, mode='constant', cval=s_image.min(), preserve_range=True, anti_aliasing=False)
    del s_image
    s_final = np.rint(s).astype(np.uint8)



    print(name)
    print(s_final.shape)
    s_finals = seg_label(s_final, channel_dim)
    for i in range(1,channel_dim):
        dice = dice_post(labels[:, :, :, (i-1)], s_finals[:, :, :, (i-1)])
        dicetot.append(dice)

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

        print('Structure ', str(i), ' : ', round(dice, 3))


    save_nii(np.asarray(s_final, dtype=np.int8), affine, newpath)
    print(name, 'CORRECTLY SAVED IN /PRED')


    return dicetot, msetot, hdtot
