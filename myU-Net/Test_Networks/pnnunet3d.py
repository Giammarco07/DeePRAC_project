from __future__ import print_function
import torch as th
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from utils.patches import gaussian_map,gaussian_map_ddt
import matplotlib
matplotlib.use('agg')
import pylab as plt
import os
import shutil
import time
import sys
ee = sys.float_info.epsilon
from utils.utils import inv_affine, inv_affine3d
from utils.figures import ensemble, np_to_img
from utils.pre_post_children import preprocessing_c,postprocessing_c, preprocessing_c25, postprocessing_c25,preprocessing_skel,postprocessing_skel, postprocessing_c_v
from utils.pre_post_adults import preprocessing_a, postprocessing_a
from Dataset import Prepare_Test_Data_new,Prepare_Test_Data_skel
from skimage.transform import resize
from skimage.morphology import skeletonize, binary_dilation, ball
from utils.losses import compute_dtm


def test(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, label_path, test_data_path,
            data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=False):
    net1 = nets[0]
    net2 = nets[1]
    if os.path.exists(data_results + '/net_3d.pth'):
        net1.load_state_dict(torch.load(data_results + '/pnet_3d.pth'))
        net2.load_state_dict(torch.load(data_results + '/net_3d.pth'))
    print('resize: ', rsz)
    print('do seg: ', do_seg)
    print(test_data_path)
    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
    pp = []
    dices_s1 = []
    mse_s1 = []
    hd_s1 = []
    if channel_dim > 2:
        dices_s2 = []
        mse_s2 = []
        hd_s2 = []
    if channel_dim > 3:
        dices_s3 = []
        mse_s3 = []
        hd_s3 = []
    if channel_dim > 4:
        dices_s4 = []
        mse_s4 = []
        hd_s4 = []

    for i in range(len(filenames_imts)):
        p = 0
        step0 = time.time()
        if input_folder == 'children':
            if do_seg:
                n_pred, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                 channel_dim, 3, res, preprocessing, do_seg, rsz)
            else:
                n_pred, utils_for_post, x_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                             os.path.join(label_path, filenames_lbts[i]), patch_size, channel_dim,
                                                             3, res, preprocessing, do_seg, rsz)
        else:
            if channel_dim == 4:
                if do_seg:
                    n_pred, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                     channel_dim, 3, res, preprocessing, do_seg,
                                                                     input='adults')
                else:
                    n_pred, utils_for_post, x_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                 channel_dim, 3, res, preprocessing, do_seg,
                                                                 input='adults')
            else:
                if do_seg:
                    n_pred, utils_for_post, x_, y_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                     channel_dim, 3, res, preprocessing, do_seg)
                else:
                    n_pred, utils_for_post, x_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                 channel_dim, 3, res, preprocessing, do_seg)
        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        pred_3d = np.zeros((channel_dim, imgshape[0], imgshape[1], imgshape[2]), dtype=np.float32)


        if do_seg:
            test_loader = Prepare_Test_Data_new(patch_ids, x_, patch_size[0], patch_size[1], patch_size[2],
                                                batch_size,
                                                workers, do_seg=do_seg, rsz=False, segt=y_)
        else:
            test_loader = Prepare_Test_Data_new(patch_ids, x_, patch_size[0], patch_size[1], patch_size[2],
                                                batch_size,
                                                workers, do_seg=do_seg, rsz=False)
        net1.eval()
        net2.eval()
        gaussian_importance_map = th.as_tensor(gaussian_map(patch_size), dtype=th.float).to(device)
        print('preprocessing of', filenames_imts[i], ' DONE')
        print('tta: ', tta)

        h = 0
        ncols = 7  # number of columns in final grid of images
        nrows = 8  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')

        print('starting inference...')

        for j, data in enumerate(test_loader, 0):
            if do_seg:
                test_image, test_target = data
                test_image = test_image.to(device)
                test_target = test_target.to(device)
            else:
                test_image = data
                test_image = test_image.to(device)

            import torch.cuda.amp as amp


            with amp.autocast():
                with torch.no_grad():
                    patches = net1(test_image)
                    predict = net2(test_image)
                    pred = predict[0]


                if tta:
                    final_pred = (1 / 8) * pred
                    # flip x
                    with torch.no_grad():
                        predict = net2(th.flip(test_image, [2]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [2]))
                    # flip y
                    with torch.no_grad():
                        predict = net2(th.flip(test_image, [3]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [3]))
                    # flip z
                    with torch.no_grad():
                        predict = net2(th.flip(test_image, [4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4]))

                    # flip x,y
                    with torch.no_grad():
                        predict = net2(th.flip(test_image, [2, 3]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [3, 2]))

                    # flip x,z
                    with torch.no_grad():
                        predict = net2(th.flip(test_image, [2, 4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4, 2]))

                    # flip y,z
                    with torch.no_grad():
                        predict = net2(th.flip(test_image, [3, 4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4, 3]))

                    # flip x,y,z
                    with torch.no_grad():
                        predict = net2(th.flip(test_image, [2, 3, 4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4, 3, 2]))

                    pred = final_pred

            for bb in range(test_image.size()[0]):
                if patches[bb] == 0:
                    p += 1
                    pred[bb, :, :, :, :] = 0

                # dist = predict[1]



            # apply of gaussian filter for reconstruction
            pred[:, :] *= gaussian_importance_map



            if ((j * batch_size) + batch_size) < n_pred:
                for h, p in zip(range(j * batch_size, j * batch_size + batch_size), range(batch_size)):
                    (d, h, w) = patch_ids[h]
                    pred_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                    w:w + patch_size[2]] += pred.data.cpu().numpy()[p, :]

            else:
                for h, p in zip(range(j * batch_size, n_pred), range(batch_size)):
                    (d, h, w) = patch_ids[h]
                    pred_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                    w:w + patch_size[2]] += pred.data.cpu().numpy()[p, :]



        print('inference DONE')
        step2 = time.time()
        if input_folder == 'children':
            dicetot, msetot, hdtot = postprocessing_c(pred_3d, os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                      data_results,
                                                      filenames_lbts[i], utils_for_post, channel_dim, 3, res, rsz)
        else:
            if channel_dim == 4:
                dicetot, msetot, hdtot = postprocessing_c(pred_3d, os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                          data_results,
                                                          filenames_lbts[i], utils_for_post, channel_dim, 3, res)
            else:
                dicetot, msetot, hdtot = postprocessing_a(pred_3d, os.path.join(label_path, filenames_lbts[i]),
                                                          data_results,
                                                          filenames_lbts[i], utils_for_post, channel_dim, 3, res)
        step3 = time.time()
        dices_s1.append(dicetot[0])
        mse_s1.append(msetot[0])
        hd_s1.append(hdtot[0])
        if channel_dim > 2:
            dices_s2.append(dicetot[1])
            mse_s2.append(msetot[1])
            hd_s2.append(hdtot[1])
        if channel_dim > 3:
            dices_s3.append(dicetot[2])
            mse_s3.append(msetot[2])
            hd_s3.append(hdtot[2])
        if channel_dim > 4:
            dices_s4.append(dicetot[3])
            mse_s4.append(msetot[3])
            hd_s4.append(hdtot[3])
        pp.append(p)
        print(p)
        print('pre: ', step1 - step0)
        print('infe: ', step2 - step1)
        print('post: ', step3 - step2)

    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    print('dices_s1: ', dices_s1)
    print('mse_s1: ', mse_s1)
    print('hd_s1: ', hd_s1)
    total_dice = []
    s1dice = np.mean(dices_s1, axis=0)
    total_dice.append(s1dice)
    s1std = np.std(dices_s1, axis=0)
    total_dice.append(s1std)
    print('Structure 1: %.4f (%.4f)' % (s1dice, s1std))
    if channel_dim > 2:
        print('dices_s2: ', dices_s2)
        print('mse_s2: ', mse_s2)
        print('hd_s2: ', hd_s2)
        s2dice = np.mean(dices_s2, axis=0)
        total_dice.append(s2dice)
        s2std = np.std(dices_s2, axis=0)
        total_dice.append(s2std)
        print('Structure 2: %.4f (%.4f)' % (s2dice, s2std))
    if channel_dim > 3:
        print('dices_s3: ', dices_s3)
        print('mse_s3: ', mse_s3)
        print('hd_s3: ', hd_s3)
        s3dice = np.mean(dices_s3, axis=0)
        total_dice.append(s3dice)
        s3std = np.std(dices_s3, axis=0)
        total_dice.append(s3std)
        print('Structure 3: %.4f (%.4f)' % (s3dice, s3std))
    if channel_dim > 4:
        print('dices_s4: ', dices_s4)
        print('mse_s4: ', mse_s4)
        print('hd_s4: ', hd_s4)
        s4dice = np.mean(dices_s4, axis=0)
        total_dice.append(s4dice)
        s4std = np.std(dices_s4, axis=0)
        total_dice.append(s4std)
        print('Structure 4: %.4f (%.4f)' % (s4dice, s4std))
        
    print(pp)
    np.save(data_results + '/dice_test.npy', np.asarray(total_dice, dtype=np.float32))

    torch.cuda.empty_cache()