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


def test(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path,
            data_results, massimo, minimo, tta, res, preprocessing, device, rsz, supervision, do_seg=False):
    if network == 'nnunet3D':
        net = nets[0]
        if os.path.exists(data_results + '/net_3d.pth'):
            net.load_state_dict(torch.load(data_results + '/net_3d.pth'))
    elif network == 'stnpose-nnunet3D':
        net1 = nets[0]
        net2 = nets[1]
        net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
        net2.load_state_dict(torch.load(data_results + '/net_2.pth'))
    print('resize: ', rsz)
    print('do seg: ', do_seg)
    print('input',in_c,' output',channel_dim)
    print(test_data_path)
    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
    print('images:',filenames_imts)
    print('labels:',filenames_lbts)
    dices_s1 = []
    precision_s1 = []
    recall_s1 = []
    hd_s1 = []
    if channel_dim > 2:
        dices_s2 = []
        precision_s2 = []
        recall_s2 = []
        hd_s2 = []
    if channel_dim > 3:
        dices_s3 = []
        precision_s3 = []
        recall_s3 = []
        hd_s3 = []
    if channel_dim > 4:
        dices_s4 = []
        precision_s4 = []
        recall_s4 = []
        hd_s4 = []

    if supervision == 'ddt-gar':
        # GEOMETRY-AWARE REFINEMENT
        print('GEOMETRY-AWARE REFINEMENT: preparing soften ball')
        list_kernel = []
        str = channel_dim - 1
        k=23
        for radius in range(1, k):
            print(radius)
            kernel = torch.as_tensor(np.repeat(np.expand_dims(ball(radius), 0)[np.newaxis, ...], str, axis=0),
                                     dtype=torch.float16).to(device)
            gaussian_gar = torch.as_tensor(
                np.repeat(np.expand_dims(gaussian_map_ddt(kernel[0, 0].size(), radius), 0)[np.newaxis, ...], str, axis=0),
                dtype=torch.float16).to(device)
            kernel = gaussian_gar * kernel
            list_kernel.append(kernel)
            del kernel, gaussian_gar

    for i in range(0,len(filenames_imts),in_c):
        step0 = time.time()
        if input_folder == 'children':
            if do_seg:
                n_pred, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                 channel_dim, in_c, 3, res, preprocessing, do_seg, rsz)
            else:
                n_pred, utils_for_post, x_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                             os.path.join(label_path, filenames_lbts[i]), patch_size, channel_dim, in_c,
                                                             3, res, preprocessing, do_seg, rsz)
        else:
            if channel_dim == 4:
                if do_seg:
                    n_pred, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                     channel_dim, in_c, 3, res, preprocessing, do_seg,
                                                                     input='adults')
                else:
                    n_pred, utils_for_post, x_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                 channel_dim, in_c, 3, res, preprocessing, do_seg,
                                                                 input='adults')
            else:
                if do_seg:
                    n_pred, utils_for_post, x_, y_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                     channel_dim, in_c, 3, res, preprocessing, do_seg)
                else:
                    n_pred, utils_for_post, x_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                 channel_dim, in_c, 3, res, preprocessing, do_seg)
        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        pred_3d = np.zeros((channel_dim, imgshape[-3], imgshape[-2], imgshape[-1]), dtype=np.float32)
        # art_3d = np.zeros((1, imgshape[0], imgshape[1], imgshape[2]), dtype=np.float32)
        # ven_3d = np.zeros((1, imgshape[0], imgshape[1], imgshape[2]), dtype=np.float32)
        if network == 'nnunet3D':
            if do_seg:
                test_loader = Prepare_Test_Data_new(patch_ids, x_, in_c, patch_size[0], patch_size[1], patch_size[2],
                                                    batch_size,
                                                    workers, do_seg=do_seg,segt=y_)
            else:
                test_loader = Prepare_Test_Data_new(patch_ids, x_, in_c, patch_size[0], patch_size[1], patch_size[2],
                                                    batch_size,
                                                    workers, do_seg=do_seg)
            net.eval()
            gaussian_importance_map = th.as_tensor(gaussian_map(patch_size), dtype=th.float).to(device)
        elif network == 'stnpose-nnunet3D':
            test_loader = Prepare_Test_Data_new(patch_ids, x_, in_c, 512, 512, 512, batch_size,
                                                workers, do_seg, rsz)
            net1.eval()
            net2.eval()
            gaussian_importance_map = th.as_tensor(gaussian_map((512, 512, 512)), dtype=th.float).to(device)
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

            if network == 'nnunet3D':
                with amp.autocast():
                    with torch.no_grad():
                        predict = net(test_image)
                        pred = predict[0]
                        if do_seg:
                            for bb in range(test_image.size()[0]):
                                if test_target[bb, 1].sum() == 0:
                                    pred[bb, 1, :, :, :] = 0
                                if test_target[bb, 2].sum() == 0:
                                    pred[bb, 2, :, :, :] = 0
                        # gt_dis = compute_dtm(test_target[:,1:].cpu().numpy(), predict[1].shape)


                        # ddt = predict[1]
                        # art = torch.argmax(ddt[:, 1:k], dim=1,keepdim=True).type(torch.float)
                        # ven = torch.argmax(ddt[:, k:(2*k-1)], dim=1,keepdim=True).type(torch.float)

                        if supervision=='ddt-gar':
                            k = 24
                            #GEOMETRY-AWARE REFINEMENT
                            print('GEOMETRY-AWARE REFINEMENT: applying soften ball...')
                            ys, yv = torch.zeros_like(pred,dtype= torch.float16).to(device), torch.zeros_like(pred[:,1:],dtype= torch.float16).to(device)
                            ddt = predict[1]
                            art = torch.argmax(ddt[:,1:k+1],dim=1) #K=23
                            ven = torch.argmax(ddt[:,k+1:(2*k+1)],dim=1)
                            skel = pred[:,1:] > 0.9
                            for radius in range(1,k):
                                 print(radius)
                                 kernel = list_kernel[radius-1]
                                 yv[:,0] = art == (radius-1)
                                 yv[:,1] = ven == (radius-1)
                                 ys[:,1:].add_(torch.clamp(torch.nn.functional.conv3d(skel*yv, kernel, padding=radius, groups=str), 0, 1))
                                 del kernel

                            ys[:, 1:] = torch.clamp(ys[:, 1:], 0, 1)
                            ys[:,0] = 1 - ys[:,1] - ys[:,2]
                            #del gaussian_gar#, cu

                            pred *= ys


                if tta:
                    final_pred = (1 / 8) * pred
                    # flip x
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [2]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [2]))
                    # flip y
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [3]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [3]))
                    # flip z
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4]))

                    # flip x,y
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [2, 3]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [3, 2]))

                    # flip x,z
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [2, 4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4, 2]))

                    # flip y,z
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [3, 4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4, 3]))

                    # flip x,y,z
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [2, 3, 4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4, 3, 2]))

                    pred = final_pred

                # dist = predict[1]

            elif network == 'stnpose-nnunet3D':
                white = torch.ones((test_image.size()[0], 1, patch_size[0], patch_size[1], patch_size[2]),
                                   requires_grad=True, dtype=torch.float16).to(device)
                black = torch.zeros((test_image.size()[0], 1, patch_size[0], patch_size[1], patch_size[2]),
                                    requires_grad=True, dtype=torch.float16).to(device)
                pose = test_image.clone().type(torch.float16)
                pose = torch.where(pose >= (pose[0].min() + 0.01),
                                   torch.where(pose < (pose[0].min() + 0.01), pose, white),
                                   black)
                with torch.no_grad():
                    x_affine, theta = net1(torch.cat((test_image, pose), dim=1))
                with amp.autocast():
                    with torch.no_grad():
                        output_affine = net2(x_affine[:, 0, :, :, :].unsqueeze(1))

                pred_ = inv_affine3d(output_affine[0], theta)

                zero = pred_[:, 0, :, :]
                zero[zero == 0] = 1
                pred_[:, 0, :, :] = zero

                pred = F.upsample_nearest(pred_, [512, 512, 512])

            # apply of gaussian filter for reconstruction
            pred[:, :] *= gaussian_importance_map
            # art[:, :] *= gaussian_importance_map
            # ven[:, :] *= gaussian_importance_map

            '''
            if do_seg:
                if j > 0 and h < 8:
                    test_target_ = torch.argmax(test_target,axis=1)
                    pred__ = torch.argmax(pred,axis=1)
                    img = np_to_img(test_image[0, 0, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                    dist1 = np_to_img(dist[0, 0, 0, :, :].data.cpu().numpy(), 'image')
                    dist2 = np_to_img(dist[0, 1, 0, :, :].data.cpu().numpy(), 'image')
                    gt_dis1 = np_to_img(gt_dis[0, 0, 0, :, :], 'image')
                    gt_dis2 = np_to_img(gt_dis[0, 1, 0, :, :], 'image')
                    prd = np_to_img(pred__[0, 0, :, :].data.cpu().numpy(), 'target')
                    tgt = np_to_img(test_target_[0, 0, :, :].data.cpu().numpy(), 'target')
                    # 0.0.11 Store results
                    axes[h, 0].set_title("Original Test Image")
                    axes[h, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[h, 1].set_title("Predicted")
                    axes[h, 1].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                    axes[h, 2].set_title("Reference")
                    axes[h, 2].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                    axes[h, 3].set_title("Predicted")
                    axes[h, 3].imshow(dist1, cmap='gray', vmin=0, vmax=255.)
                    axes[h, 4].set_title("Reference")
                    axes[h, 4].imshow(gt_dis1, cmap='gray', vmin=0, vmax=255.)
                    axes[h, 5].set_title("Predicted")
                    axes[h, 5].imshow(dist2, cmap='gray', vmin=0, vmax=255.)
                    axes[h, 6].set_title("Reference")
                    axes[h, 6].imshow(gt_dis2, cmap='gray', vmin=0, vmax=255.)
                    f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png', bbox_inches='tight')
                    h += 1
            '''

            if network == 'nnunet3D':
                if ((j * batch_size) + batch_size) < n_pred:
                    for h, p in zip(range(j * batch_size, j * batch_size + batch_size), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        pred_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                        w:w + patch_size[2]] += pred.data.cpu().numpy()[p, :]

                        # art_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                        # w:w + patch_size[2]] += art.data.cpu().numpy()[p, :]

                        # ven_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                        # w:w + patch_size[2]] += ven.data.cpu().numpy()[p, :]
                else:
                    for h, p in zip(range(j * batch_size, n_pred), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        pred_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                        w:w + patch_size[2]] += pred.data.cpu().numpy()[p, :]

                        # art_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                        # w:w + patch_size[2]] += art.data.cpu().numpy()[p, :]

                        # ven_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                        # w:w + patch_size[2]] += ven.data.cpu().numpy()[p, :]

            elif network == 'stnpose-nnunet3D':
                if ((j * batch_size) + batch_size) < n_pred:
                    for h, p in zip(range(j * batch_size, j * batch_size + batch_size), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        pred_3d[:, d:d + 512, h:h + 512,
                        w:w + 512] += pred.data.cpu().numpy()[p, :]
                else:
                    for h, p in zip(range(j * batch_size, n_pred), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        pred_3d[:, d:d + 512, h:h + 512,
                        w:w + 512] += pred.data.cpu().numpy()[p, :]

        print('inference DONE')
        step2 = time.time()
        if input_folder == 'children':
            dicetot, precisiontot, recalltot, hdtot = postprocessing_c(pred_3d, os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                      data_results,
                                                      filenames_lbts[i], utils_for_post, channel_dim, 3, res, rsz)
            # postprocessing_c_v(art_3d, os.path.join(label_path, filenames_lbts[i]), data_results,
            #                                       filenames_lbts[i], utils_for_post, channel_dim, 3, res, v='art')
            # postprocessing_c_v(ven_3d, os.path.join(label_path, filenames_lbts[i]), data_results,
            #                                       filenames_lbts[i], utils_for_post, channel_dim, 3, res, v='ven')
        else:
            if channel_dim == 4:
                dicetot, precisiontot, recalltot, hdtot = postprocessing_c(pred_3d, os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                          data_results,
                                                          filenames_lbts[i], utils_for_post, channel_dim, 3, res)
            else:
                dicetot, precisiontot, recalltot, hdtot = postprocessing_a(pred_3d, os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                          data_results,
                                                          filenames_lbts[i], utils_for_post, channel_dim, 3, res)
        step3 = time.time()
        dices_s1.append(dicetot[0])
        precision_s1.append(precisiontot[0])
        recall_s1.append(recalltot[0])
        hd_s1.append(hdtot[0])
        if channel_dim > 2:
            dices_s2.append(dicetot[1])
            precision_s2.append(precisiontot[1])
            recall_s2.append(recalltot[1])
            hd_s2.append(hdtot[1])
        if channel_dim > 3:
            dices_s3.append(dicetot[2])
            precision_s3.append(precisiontot[2])
            recall_s3.append(recalltot[2])
            hd_s3.append(hdtot[2])
        if channel_dim > 4:
            dices_s4.append(dicetot[3])
            precision_s4.append(precisiontot[3])
            recall_s4.append(recalltot[3])
            hd_s4.append(hdtot[3])
        print('pre: ', step1 - step0)
        print('infe: ', step2 - step1)
        print('post: ', step3 - step2)

    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    print('dices_s1: ', dices_s1)
    print('precision_s1: ', precision_s1)
    print('recall_s1: ', recall_s1)
    print('hd_s1: ', hd_s1)
    total_dice = []
    s1dice = np.mean(dices_s1, axis=0)
    total_dice.append(s1dice)
    s1std = np.std(dices_s1, axis=0)
    total_dice.append(s1std)
    np.savetxt(data_results + '/s1.csv', (dices_s1, precision_s1, recall_s1, hd_s1), fmt='%f', delimiter=',')
    print('Structure 1: %.4f (%.4f)' % (s1dice, s1std))
    if channel_dim > 2:
        print('dices_s2: ', dices_s2)
        print('precision_s2: ', precision_s2)
        print('recall_s2: ', recall_s2)
        print('hd_s2: ', hd_s2)
        s2dice = np.mean(dices_s2, axis=0)
        total_dice.append(s2dice)
        s2std = np.std(dices_s2, axis=0)
        total_dice.append(s2std)
        np.savetxt(data_results + '/s2.csv', (dices_s2, precision_s2, recall_s2, hd_s2), fmt='%f', delimiter=',')
        print('Structure 2: %.4f (%.4f)' % (s2dice, s2std))
    if channel_dim > 3:
        print('dices_s3: ', dices_s3)
        print('precision_s3: ', precision_s3)
        print('recall_s3: ', recall_s3)
        print('hd_s3: ', hd_s3)
        s3dice = np.mean(dices_s3, axis=0)
        total_dice.append(s3dice)
        s3std = np.std(dices_s3, axis=0)
        total_dice.append(s3std)
        np.savetxt(data_results + '/3.csv', (dices_s3, precision_s3, recall_s3, hd_s3), fmt='%f', delimiter=',')
        print('Structure 3: %.4f (%.4f)' % (s3dice, s3std))
    if channel_dim > 4:
        print('dices_s4: ', dices_s4)
        print('precision_s4: ', precision_s4)
        print('recall_s4: ', recall_s4)
        print('hd_s4: ', hd_s4)
        s4dice = np.mean(dices_s4, axis=0)
        total_dice.append(s4dice)
        s4std = np.std(dices_s4, axis=0)
        total_dice.append(s4std)
        np.savetxt(data_results + '/s4.csv', (dices_s4, precision_s4, recall_s4, hd_s4), fmt='%f', delimiter=',')
        print('Structure 4: %.4f (%.4f)' % (s4dice, s4std))

    np.save(data_results + '/dice_test.npy', np.asarray(total_dice, dtype=np.float32))
    torch.cuda.empty_cache()


def test_skel(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, label_path, test_data_path,
                data_results, massimo, minimo, tta, res, preprocessing, device, rsz, norm):
    if network == 'nnunet3D':
        net = nets[0]
        if os.path.exists(data_results + '/net_3d.pth'):
            net.load_state_dict(torch.load(data_results + '/net_3d.pth'))
    elif network == 'stnpose-nnunet3D':
        net1 = nets[0]
        net2 = nets[1]
        net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
        net2.load_state_dict(torch.load(data_results + '/net_2.pth'))
    print('resize: ', rsz)
    print(test_data_path)
    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
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
        step0 = time.time()
        if input_folder == 'children':
            n_pred, utils_for_post, x_, y_ = preprocessing_skel(os.path.join(test_data_path, filenames_imts[i]),
                                                                os.path.join(label_path, filenames_lbts[i]),
                                                                channel_dim, 3, res, preprocessing, rsz)

        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        pred_3d = np.zeros((channel_dim, imgshape[0], imgshape[1], imgshape[2]), dtype=np.float32)
        if network == 'nnunet3D':
            test_loader = Prepare_Test_Data_skel(patch_ids, x_, patch_size[0], patch_size[1], patch_size[2], batch_size,
                                                 workers, rsz=False, segt=y_, norm=norm)
            net.eval()
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

            test_image, test_target = data
            test_image = test_image.to(device)
            test_target = test_target.to(device)

            import torch.cuda.amp as amp

            if network == 'nnunet3D':
                with amp.autocast():
                    with torch.no_grad():
                        predict = net(test_image)
                        pred = predict[0]
                        # gt_dis = compute_dtm(test_target[:,1:].cpu().numpy(), predict[1].shape)

                if tta:
                    final_pred = (1 / 8) * pred
                    # flip x
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [2]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [2]))
                    # flip y
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [3]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [3]))
                    # flip z
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4]))

                    # flip x,y
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [2, 3]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [3, 2]))

                    # flip x,z
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [2, 4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4, 2]))

                    # flip y,z
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [3, 4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4, 3]))

                    # flip x,y,z
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [2, 3, 4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4, 3, 2]))

                    pred = final_pred

                # dist = predict[1]

            elif network == 'stnpose-nnunet3D':
                white = torch.ones((test_image.size()[0], 1, patch_size[0], patch_size[1], patch_size[2]),
                                   requires_grad=True, dtype=torch.float16).to(device)
                black = torch.zeros((test_image.size()[0], 1, patch_size[0], patch_size[1], patch_size[2]),
                                    requires_grad=True, dtype=torch.float16).to(device)
                pose = test_image.clone().type(torch.float16)
                pose = torch.where(pose >= (pose[0].min() + 0.01),
                                   torch.where(pose < (pose[0].min() + 0.01), pose, white),
                                   black)
                with torch.no_grad():
                    x_affine, theta = net1(torch.cat((test_image, pose), dim=1))
                with amp.autocast():
                    with torch.no_grad():
                        output_affine = net2(x_affine[:, 0, :, :, :].unsqueeze(1))

                pred_ = inv_affine3d(output_affine[0], theta)

                zero = pred_[:, 0, :, :]
                zero[zero == 0] = 1
                pred_[:, 0, :, :] = zero

                pred = F.upsample_nearest(pred_, [512, 512, 512])

            # apply of gaussian filter for reconstruction
            pred[:, :] *= gaussian_importance_map

            if network == 'nnunet3D':
                if ((j * batch_size) + batch_size) < n_pred:
                    for h, p in zip(range(j * batch_size, j * batch_size + batch_size), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        patch = pred_3d[0, d - 16:d + 16, h - 32:h + 32, w - 32:w + 32]
                        x, y, z = 16 - patch.shape[0] // 2, 32 - patch.shape[1] // 2, 32 - patch.shape[2] // 2
                        pred_3d[:, d - 16:d + 16, h - 32:h + 32, w - 32:w + 32] += pred.data.cpu().numpy()[p, :,
                                                                                   x:x + patch.shape[0],
                                                                                   y:y + patch.shape[1],
                                                                                   z:z + patch.shape[2]]
                else:
                    for h, p in zip(range(j * batch_size, n_pred), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        patch = pred_3d[0, d - 16:d + 16, h - 32:h + 32, w - 32:w + 32]
                        x, y, z = 16 - patch.shape[0] // 2, 32 - patch.shape[1] // 2, 32 - patch.shape[2] // 2
                        pred_3d[:, d - 16:d + 16, h - 32:h + 32, w - 32:w + 32] += pred.data.cpu().numpy()[p, :,
                                                                                   x:x + patch.shape[0],
                                                                                   y:y + patch.shape[1],
                                                                                   z:z + patch.shape[2]]
            elif network == 'stnpose-nnunet3D':
                if ((j * batch_size) + batch_size) < n_pred:
                    for h, p in zip(range(j * batch_size, j * batch_size + batch_size), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        pred_3d[:, d:d + 512, h:h + 512,
                        w:w + 512] += pred.data.cpu().numpy()[p, :]
                else:
                    for h, p in zip(range(j * batch_size, n_pred), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        pred_3d[:, d:d + 512, h:h + 512,
                        w:w + 512] += pred.data.cpu().numpy()[p, :]

        print('inference DONE')
        step2 = time.time()
        if input_folder == 'children':
            dicetot, msetot, hdtot = postprocessing_skel(pred_3d, os.path.join(label_path, filenames_lbts[i]),
                                                         data_results,
                                                         filenames_lbts[i], utils_for_post, channel_dim, 3, res, rsz)

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

    np.save(data_results + '/dice_test.npy', np.asarray(total_dice, dtype=np.float32))

    torch.cuda.empty_cache()

def test_cascade(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, label_path, test_data_path,
            data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=False):
    if network == 'nnunet3D':
        net = nets[0]
        net_old = nets[1]
        if os.path.exists(data_results + '/net_3d.pth'):
            net.load_state_dict(torch.load(data_results + '/net_3d.pth'))
    elif network == 'stnpose-nnunet3D':
        net1 = nets[0]
        net2 = nets[1]
        net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
        net2.load_state_dict(torch.load(data_results + '/net_2.pth'))
    print('resize: ', rsz)
    print('do seg: ', do_seg)
    print(test_data_path)
    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
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

    '''
    # GEOMETRY-AWARE REFINEMENT
    print('GEOMETRY-AWARE REFINEMENT: preparing soften ball')
    list_kernel = []
    str = channel_dim - 1
    k=23
    for radius in range(1, k):
        print(radius)
        kernel = torch.as_tensor(np.repeat(np.expand_dims(ball(radius), 0)[np.newaxis, ...], str, axis=0),
                                 dtype=torch.float16).to(device)
        gaussian_gar = torch.as_tensor(
            np.repeat(np.expand_dims(gaussian_map_ddt(kernel[0, 0].size(), radius), 0)[np.newaxis, ...], str, axis=0),
            dtype=torch.float16).to(device)
        kernel = gaussian_gar * kernel
        list_kernel.append(kernel)
        del kernel, gaussian_gar
    '''

    for i in range(len(filenames_imts)):
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
        # art_3d = np.zeros((1, imgshape[0], imgshape[1], imgshape[2]), dtype=np.float32)
        # ven_3d = np.zeros((1, imgshape[0], imgshape[1], imgshape[2]), dtype=np.float32)
        if network == 'nnunet3D':
            if do_seg:
                test_loader = Prepare_Test_Data_new(patch_ids, x_, patch_size[0], patch_size[1], patch_size[2],
                                                    batch_size,
                                                    workers, do_seg=do_seg, rsz=False, segt=y_)
            else:
                test_loader = Prepare_Test_Data_new(patch_ids, x_, patch_size[0], patch_size[1], patch_size[2],
                                                    batch_size,
                                                    workers, do_seg=do_seg, rsz=False)
            net.eval()
            gaussian_importance_map = th.as_tensor(gaussian_map(patch_size), dtype=th.float).to(device)
        elif network == 'stnpose-nnunet3D':
            test_loader = Prepare_Test_Data_new(patch_ids, x_, 512, 512, 512, batch_size,
                                                workers, do_seg, rsz)
            net1.eval()
            net2.eval()
            gaussian_importance_map = th.as_tensor(gaussian_map((512, 512, 512)), dtype=th.float).to(device)
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

            if network == 'nnunet3D':

                with amp.autocast():
                    with torch.no_grad():
                        test_ref = net_old(test_image)
                        test_refe = torch.argmax(test_ref[0], dim=1, keepdim=True)
                test_input = torch.cat((test_image, test_refe), 1)

                with amp.autocast():
                    with torch.no_grad():
                        predict = net(test_input)
                        pred = predict[0]
                        if do_seg:
                            for bb in range(test_image.size()[0]):
                                if test_target[bb, 1].sum() == 0:
                                    pred[bb, 1, :, :, :] = 0
                                if test_target[bb, 2].sum() == 0:
                                    pred[bb, 2, :, :, :] = 0
                        # gt_dis = compute_dtm(test_target[:,1:].cpu().numpy(), predict[1].shape)

                        # k=24
                        # ddt = predict[1]
                        # art = torch.argmax(ddt[:, 1:k], dim=1,keepdim=True).type(torch.float)
                        # ven = torch.argmax(ddt[:, k:(2*k-1)], dim=1,keepdim=True).type(torch.float)

                        '''
                        #GEOMETRY-AWARE REFINEMENT
                        print('GEOMETRY-AWARE REFINEMENT: applying soften ball...')
                        ys, yv = torch.zeros_like(pred,dtype= torch.float16).to(device), torch.zeros_like(pred[:,1:],dtype= torch.float16).to(device)
                        ddt = predict[1]
                        art = torch.argmax(ddt[:,1:k+1],dim=1) #K=23
                        ven = torch.argmax(ddt[:,k+1:2k+1)],dim=1)
                        skel = pred[:,1:] > 0.9
                        for radius in range(1,k):
                             print(radius)
                             kernel = list_kernel[radius-1]
                             yv[:,0] = art == (radius-1)
                             yv[:,1] = ven == (radius-1)
                             ys[:,1:].add_(torch.clamp(torch.nn.functional.conv3d(skel*yv, kernel, padding=radius, groups=str), 0, 1))
                             del kernel

                        ys[:, 1:] = torch.clamp(ys[:, 1:], 0, 1)
                        ys[:,0] = 1 - ys[:,1] - ys[:,2]
                        #del gaussian_gar#, cu

                        pred *= ys
                        '''

                if tta:
                    final_pred = (1 / 8) * pred
                    # flip x
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [2]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [2]))
                    # flip y
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [3]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [3]))
                    # flip z
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4]))

                    # flip x,y
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [2, 3]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [3, 2]))

                    # flip x,z
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [2, 4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4, 2]))

                    # flip y,z
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [3, 4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4, 3]))

                    # flip x,y,z
                    with torch.no_grad():
                        predict = net(th.flip(test_image, [2, 3, 4]))
                        pred = predict[0]
                    final_pred += (1 / 8) * (th.flip(pred, [4, 3, 2]))

                    pred = final_pred

                # dist = predict[1]

            elif network == 'stnpose-nnunet3D':
                white = torch.ones((test_image.size()[0], 1, patch_size[0], patch_size[1], patch_size[2]),
                                   requires_grad=True, dtype=torch.float16).to(device)
                black = torch.zeros((test_image.size()[0], 1, patch_size[0], patch_size[1], patch_size[2]),
                                    requires_grad=True, dtype=torch.float16).to(device)
                pose = test_image.clone().type(torch.float16)
                pose = torch.where(pose >= (pose[0].min() + 0.01),
                                   torch.where(pose < (pose[0].min() + 0.01), pose, white),
                                   black)
                with torch.no_grad():
                    x_affine, theta = net1(torch.cat((test_image, pose), dim=1))
                with amp.autocast():
                    with torch.no_grad():
                        output_affine = net2(x_affine[:, 0, :, :, :].unsqueeze(1))

                pred_ = inv_affine3d(output_affine[0], theta)

                zero = pred_[:, 0, :, :]
                zero[zero == 0] = 1
                pred_[:, 0, :, :] = zero

                pred = F.upsample_nearest(pred_, [512, 512, 512])

            # apply of gaussian filter for reconstruction
            pred[:, :] *= gaussian_importance_map
            # art[:, :] *= gaussian_importance_map
            # ven[:, :] *= gaussian_importance_map

            '''
            if do_seg:
                if j > 0 and h < 8:
                    test_target_ = torch.argmax(test_target,axis=1)
                    pred__ = torch.argmax(pred,axis=1)
                    img = np_to_img(test_image[0, 0, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                    dist1 = np_to_img(dist[0, 0, 0, :, :].data.cpu().numpy(), 'image')
                    dist2 = np_to_img(dist[0, 1, 0, :, :].data.cpu().numpy(), 'image')
                    gt_dis1 = np_to_img(gt_dis[0, 0, 0, :, :], 'image')
                    gt_dis2 = np_to_img(gt_dis[0, 1, 0, :, :], 'image')
                    prd = np_to_img(pred__[0, 0, :, :].data.cpu().numpy(), 'target')
                    tgt = np_to_img(test_target_[0, 0, :, :].data.cpu().numpy(), 'target')
                    # 0.0.11 Store results
                    axes[h, 0].set_title("Original Test Image")
                    axes[h, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[h, 1].set_title("Predicted")
                    axes[h, 1].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                    axes[h, 2].set_title("Reference")
                    axes[h, 2].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                    axes[h, 3].set_title("Predicted")
                    axes[h, 3].imshow(dist1, cmap='gray', vmin=0, vmax=255.)
                    axes[h, 4].set_title("Reference")
                    axes[h, 4].imshow(gt_dis1, cmap='gray', vmin=0, vmax=255.)
                    axes[h, 5].set_title("Predicted")
                    axes[h, 5].imshow(dist2, cmap='gray', vmin=0, vmax=255.)
                    axes[h, 6].set_title("Reference")
                    axes[h, 6].imshow(gt_dis2, cmap='gray', vmin=0, vmax=255.)
                    f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png', bbox_inches='tight')
                    h += 1
            '''

            if network == 'nnunet3D':
                if ((j * batch_size) + batch_size) < n_pred:
                    for h, p in zip(range(j * batch_size, j * batch_size + batch_size), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        pred_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                        w:w + patch_size[2]] += pred.data.cpu().numpy()[p, :]

                        # art_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                        # w:w + patch_size[2]] += art.data.cpu().numpy()[p, :]

                        # ven_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                        # w:w + patch_size[2]] += ven.data.cpu().numpy()[p, :]
                else:
                    for h, p in zip(range(j * batch_size, n_pred), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        pred_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                        w:w + patch_size[2]] += pred.data.cpu().numpy()[p, :]

                        # art_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                        # w:w + patch_size[2]] += art.data.cpu().numpy()[p, :]

                        # ven_3d[:, d:d + patch_size[0], h:h + patch_size[1],
                        # w:w + patch_size[2]] += ven.data.cpu().numpy()[p, :]

            elif network == 'stnpose-nnunet3D':
                if ((j * batch_size) + batch_size) < n_pred:
                    for h, p in zip(range(j * batch_size, j * batch_size + batch_size), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        pred_3d[:, d:d + 512, h:h + 512,
                        w:w + 512] += pred.data.cpu().numpy()[p, :]
                else:
                    for h, p in zip(range(j * batch_size, n_pred), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        pred_3d[:, d:d + 512, h:h + 512,
                        w:w + 512] += pred.data.cpu().numpy()[p, :]

        print('inference DONE')
        step2 = time.time()
        if input_folder == 'children':
            dicetot, msetot, hdtot = postprocessing_c(pred_3d, os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                      data_results,
                                                      filenames_lbts[i], utils_for_post, channel_dim, 3, res, rsz)
            # postprocessing_c_v(art_3d, os.path.join(label_path, filenames_lbts[i]), data_results,
            #                                       filenames_lbts[i], utils_for_post, channel_dim, 3, res, v='art')
            # postprocessing_c_v(ven_3d, os.path.join(label_path, filenames_lbts[i]), data_results,
            #                                       filenames_lbts[i], utils_for_post, channel_dim, 3, res, v='ven')
        else:
            if channel_dim == 4:
                dicetot, msetot, hdtot = postprocessing_c(pred_3d, os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                          data_results,
                                                          filenames_lbts[i], utils_for_post, channel_dim, 3, res)
            else:
                dicetot, msetot, hdtot = postprocessing_a(pred_3d, os.path.join(label_path, filenames_lbts[i]), patch_size,
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

    np.save(data_results + '/dice_test.npy', np.asarray(total_dice, dtype=np.float32))

    torch.cuda.empty_cache()


def val(input_folder, patch_size, batch_size, workers, net, channel_dim, in_c, label_path, test_data_path,
         data_results, massimo, minimo, tta, res, preprocessing, device, epoch, do_seg=True):
    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
    dices = np.zeros((len(filenames_imts),(channel_dim-1)))
    hd = np.zeros((len(filenames_imts),(channel_dim-1)))
    batch_size += batch_size//2

    for i in range(0, len(filenames_imts), in_c):
        if input_folder == 'children':
            if do_seg:
                n_pred, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]),
                                                                 patch_size,
                                                                 channel_dim, in_c, 3, res, preprocessing, do_seg)
            else:
                n_pred, utils_for_post, x_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                             os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                             channel_dim, in_c,
                                                             3, res, preprocessing, do_seg)
        else:
            if channel_dim == 4:
                if do_seg:
                    n_pred, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]),
                                                                     patch_size,
                                                                     channel_dim, in_c, 3, res, preprocessing, do_seg,
                                                                     input='adults')
                else:
                    n_pred, utils_for_post, x_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]),
                                                                 patch_size,
                                                                 channel_dim, in_c, 3, res, preprocessing, do_seg,
                                                                 input='adults')
            else:
                if do_seg:
                    n_pred, utils_for_post, x_, y_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]),
                                                                     patch_size,
                                                                     channel_dim, 3, res, preprocessing, do_seg)
                else:
                    n_pred, utils_for_post, x_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]),
                                                                 patch_size,
                                                                 channel_dim, 3, res, preprocessing, do_seg)
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        pred_3d = np.zeros((channel_dim, imgshape[-3], imgshape[-2], imgshape[-1]), dtype=np.float32)
        if do_seg:
            test_loader = Prepare_Test_Data_new(patch_ids, x_, in_c, patch_size[0], patch_size[1], patch_size[2],
                                                batch_size,
                                                workers, do_seg=do_seg, rsz=False, segt=y_)
        else:
            test_loader = Prepare_Test_Data_new(patch_ids, x_, in_c, patch_size[0], patch_size[1], patch_size[2],
                                                batch_size,
                                                workers, do_seg=do_seg, rsz=False)
        net.eval()
        gaussian_importance_map = th.as_tensor(gaussian_map(patch_size), dtype=th.float).to(device)
        print('preprocessing of', filenames_imts[i], ' DONE')
        print('tta: ', tta)

        hh = 0
        ncols = 3  # number of columns in final grid of images
        nrows = 8  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')

        print('starting validation...')

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
                    predict = net(test_image)
                    pred = predict[0]

            if tta:
                final_pred = (1 / 8) * pred
                # flip x
                with torch.no_grad():
                    predict = net(th.flip(test_image, [2]))
                    pred = predict[0]
                final_pred += (1 / 8) * (th.flip(pred, [2]))
                # flip y
                with torch.no_grad():
                    predict = net(th.flip(test_image, [3]))
                    pred = predict[0]
                final_pred += (1 / 8) * (th.flip(pred, [3]))
                # flip z
                with torch.no_grad():
                    predict = net(th.flip(test_image, [4]))
                    pred = predict[0]
                final_pred += (1 / 8) * (th.flip(pred, [4]))

                # flip x,y
                with torch.no_grad():
                    predict = net(th.flip(test_image, [2, 3]))
                    pred = predict[0]
                final_pred += (1 / 8) * (th.flip(pred, [3, 2]))

                # flip x,z
                with torch.no_grad():
                    predict = net(th.flip(test_image, [2, 4]))
                    pred = predict[0]
                final_pred += (1 / 8) * (th.flip(pred, [4, 2]))

                # flip y,z
                with torch.no_grad():
                    predict = net(th.flip(test_image, [3, 4]))
                    pred = predict[0]
                final_pred += (1 / 8) * (th.flip(pred, [4, 3]))

                # flip x,y,z
                with torch.no_grad():
                    predict = net(th.flip(test_image, [2, 3, 4]))
                    pred = predict[0]
                final_pred += (1 / 8) * (th.flip(pred, [4, 3, 2]))

                pred = final_pred


            # apply of gaussian filter for reconstruction
            pred[:, :] *= gaussian_importance_map

            if do_seg==True and (epoch%25==0 or epoch+1==1000):
                if j%10==0 and hh < 8:
                    test_target_ = torch.argmax(test_target,axis=1)
                    pred__ = torch.argmax(pred,axis=1)
                    img = np_to_img(test_image[0, 0, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                    prd = np_to_img(pred__[0, 0, :, :].data.cpu().numpy(), 'target')
                    tgt = np_to_img(test_target_[0, 0, :, :].data.cpu().numpy(), 'target')
                    # 0.0.11 Store results
                    axes[hh, 0].set_title("Original Test Image")
                    axes[hh, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[hh, 1].set_title("Predicted")
                    axes[hh, 1].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                    axes[hh, 2].set_title("Reference")
                    axes[hh, 2].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                    hh += 1
                    if hh==8:
                      f.savefig(data_results + '/images_val_' + str(epoch) + '_' + filenames_imts[i][-20:-11] + '.png', bbox_inches='tight')
                      del f


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


        print('validation DONE')
        if input_folder == 'children':
            dicetot, _, _, hdtot = postprocessing_c(pred_3d,
                                                                       os.path.join(label_path, filenames_lbts[i]),
                                                                       patch_size,
                                                                       data_results,
                                                                       filenames_lbts[i], utils_for_post, channel_dim,
                                                                       3, res,save=False)
        else:
            if channel_dim == 4:
                dicetot, _, _, hdtot = postprocessing_c(pred_3d,
                                                                           os.path.join(label_path, filenames_lbts[i]),
                                                                           patch_size,
                                                                           data_results,
                                                                           filenames_lbts[i], utils_for_post,
                                                                           channel_dim, 3, res)
            else:
                dicetot, _, hdtot = postprocessing_a(pred_3d, os.path.join(label_path, filenames_lbts[i]),
                                                          patch_size,
                                                          data_results,
                                                          filenames_lbts[i], utils_for_post, channel_dim, 3, res)
        dices[i,:] = dicetot
        hd[i,:] = hdtot

    print('dices_s1: ', dices)
    print('hd_s1: ', hd)

    return np.asarray(dices, dtype=np.float32),np.asarray(hd, dtype=np.float32)
