from __future__ import print_function
import torch as th
import torch.nn.parallel
import torch.utils.data
import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as plt
import os
import time
import sys
ee = sys.float_info.epsilon
from utils.utils import inv_affine
from utils.figures import ensemble, np_to_img
from utils.pre_post_children import preprocessing_c,postprocessing_c
from utils.pre_post_adults import preprocessing_a, postprocessing_a
from Dataset import Prepare_Test_Data_new
import torch.nn.functional as F
def test(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing, device, do_seg=True, rsz=False):
    if network == 'nnunet2D':
        net = nets[0]
        net.load_state_dict(torch.load(data_results + '/net.pth'))
    elif network == 'stnpose-nnunet2D' or network == 'stncrop-nnunet2D' or network == 'stncrop3-nnunet2D':
        net1 = nets[0]
        net2 = nets[1]
        net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
        net2.load_state_dict(torch.load(data_results + '/net_2.pth'))
    elif network=='stnposennunet2D':
        net1 = nets[0]
        net2 = nets[1]
        net1.load_state_dict(torch.load(data_results + '/net_pose.pth'))
        net2.load_state_dict(torch.load(data_results + '/net.pth'))
    elif network=='stnposecropnnunet2D':
        net1 = nets[0]
        net2 = nets[1]
        net3 = nets[2]
        net1.load_state_dict(torch.load(data_results + '/net_pose.pth'))
        net2.load_state_dict(torch.load(data_results + '/net_crop.pth'))
        net3.load_state_dict(torch.load(data_results + '/net.pth'))
    else:
        net1 = nets[0]
        net2 = nets[1]
        net3 = nets[2]
        net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
        net2.load_state_dict(torch.load(data_results + '/net_2.pth'))
        net3.load_state_dict(torch.load(data_results + '/net_3.pth'))

    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
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
        mse_s3 = []
        hd_s3 = []

    for i in range(len(filenames_imts)):
        step0 = time.time()
        if input_folder == 'children':
            if do_seg:
                n_pred, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                 channel_dim, in_c, 2, res, preprocessing, do_seg,rsz)
            else:
                n_pred, utils_for_post, x_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                             os.path.join(label_path, filenames_lbts[i]), patch_size, channel_dim,in_c, 
                                                             2, res, preprocessing, do_seg,rsz)
        else:
            if do_seg:
                n_pred, utils_for_post, x_, y_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                 channel_dim, in_c, 2, res, preprocessing, do_seg,rsz)
            else:
                n_pred, utils_for_post, x_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                             os.path.join(label_path, filenames_lbts[i]), patch_size, channel_dim,in_c, 
                                                             2, res, preprocessing, do_seg,rsz)
        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        pred_3d = np.zeros((channel_dim, imgshape[-3], imgshape[-2], imgshape[-1]), dtype=np.float32)
        test_loader = Prepare_Test_Data_new(patch_ids, x_, in_c, 1, patch_size[0], patch_size[1], batch_size,
                                                workers,do_seg=do_seg,segt=y_)
        print('preprocessing of', filenames_imts[i], ' DONE')
        print('tta: ', tta)

        if network == 'nnunet2D':
            net.eval()
            ncols = 3  # number of columns in final grid of images
        elif network == 'stnpose-nnunet2D' or network == 'stncrop-nnunet2D' or network == 'stncrop3-nnunet2D'or network=='stnposennunet2D':
            net1.eval()
            net2.eval()
            ncols = 5  # number of columns in final grid of images
        else:
            net1.eval()
            net2.eval()
            net3.eval()
            ncols = 7  # number of columns in final grid of images

        g = 0
        nrows = 5  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')
        print('starting inference...')
        for j, data in enumerate(test_loader, 0):

            if do_seg:
                test_image, test_target = data
                test_image = test_image.squeeze(2).to(device)
                test_target = test_target.squeeze(2).to(device)
            else:
                test_image = data
                test_image = test_image.squeeze(2).to(device)

            if network == 'nnunet2D':
                with torch.no_grad():
                    pred, _, _, _, _, _ = net(test_image)

                    if tta:
                        final_pred = (1 / 4) * pred
                        # flip x
                        with torch.no_grad():
                            pred, _, _, _, _, _ = net(th.flip(test_image, [2]))
                        final_pred += (1 / 4) * (th.flip(pred, [2]))
                        # flip y
                        with torch.no_grad():
                            pred, _, _, _, _, _ = net(th.flip(test_image, [3]))
                        final_pred += (1 / 4) * (th.flip(pred, [3]))

                        # flip x,y
                        with torch.no_grad():
                            pred, _, _, _, _, _ = net(th.flip(test_image, [2, 3]))
                        final_pred += (1 / 4) * (th.flip(pred, [2, 3]))

                        pred = final_pred

            elif network == 'stnpose-nnunet2D'or network=='stnposennunet2D':
                white = torch.ones((test_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
                black = torch.zeros((test_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
                pose = test_image.clone()
                pose = torch.where(pose >= (test_image[0].min() + 0.01),
                                   torch.where(pose < (test_image[0].min() + 0.01), pose, white), black)
                with torch.no_grad():
                    x_affine, theta = net1(torch.cat((test_image, pose), dim=1))#, padding='zeros')
                    #x_affine[x_affine == 0] = test_image[0].min()
                    output_affine = net2(x_affine[:, 0, :, :].unsqueeze(1))
                    pred = inv_affine(output_affine[0], theta)
                    zero = pred[:, 0, :, :]
                    zero[zero == 0] = 1
                    pred[:, 0, :, :] = zero

                if tta:
                    final_pred = (1 / 4) * pred
                    # flip x
                    with torch.no_grad():
                        output_affine = net2(th.flip(x_affine[:, 0, :, :].unsqueeze(1), [2]))
                        pred = inv_affine(th.flip(output_affine[0], [2]), theta)
                    final_pred += (1 / 4) * (pred)

                    # flip y
                    with torch.no_grad():
                        output_affine = net2(th.flip(x_affine[:, 0, :, :].unsqueeze(1), [3]))
                        pred = inv_affine(th.flip(output_affine[0], [3]), theta)
                    final_pred += (1 / 4) * (pred)

                    # flip x,y
                    with torch.no_grad():
                        output_affine = net2(th.flip(x_affine[:, 0, :, :].unsqueeze(1), [2, 3]))
                        pred = inv_affine(th.flip(output_affine[0], [2, 3]), theta)
                    final_pred += (1 / 4) * (pred)

                    pred = final_pred

            elif network == 'stncrop-nnunet2D':
                with torch.no_grad():
                    x_affine, theta = net1(test_image, mode='security')
                    output_affine = net2(x_affine)
                    pred = inv_affine(output_affine[0], theta)
                    zero = pred[:, 0, :, :]
                    zero[zero == 0] = 1
                    pred[:, 0, :, :] = zero

                if tta:
                    final_pred = (1 / 4) * pred
                    # flip x
                    with torch.no_grad():
                        output_affine = net2(th.flip(x_affine, [2]))
                        pred = inv_affine(th.flip(output_affine[0], [2]), theta)
                    final_pred += (1 / 4) * (pred)

                    # flip y
                    with torch.no_grad():
                        output_affine = net2(th.flip(x_affine, [3]))
                        pred = inv_affine(th.flip(output_affine[0], [3]), theta)
                    final_pred += (1 / 4) * (pred)

                    # flip x,y
                    with torch.no_grad():
                        output_affine = net2(th.flip(x_affine, [2, 3]))
                        pred = inv_affine(th.flip(output_affine[0], [2, 3]), theta)
                    final_pred += (1 / 4) * (pred)

                    pred = final_pred
            elif network == 'stncrop3-nnunet2D':
                with torch.no_grad():
                    x_affine, theta = net1(test_image, mode='security')
                    output_affine = net2(x_affine)
                    pred = output_affine[0]

                if tta:
                    final_pred = (1 / 4) * pred
                    # flip x
                    with torch.no_grad():
                        output_affine = net2(th.flip(x_affine, [2]))
                        pred = th.flip(output_affine[0], [2])
                    final_pred += (1 / 4) * (pred)

                    # flip y
                    with torch.no_grad():
                        output_affine = net2(th.flip(x_affine, [3]))
                        pred = th.flip(output_affine[0], [3])
                    final_pred += (1 / 4) * (pred)

                    # flip x,y
                    with torch.no_grad():
                        output_affine = net2(th.flip(x_affine, [2, 3]))
                        pred = th.flip(output_affine[0], [2, 3])
                    final_pred += (1 / 4) * (pred)

                    pred = final_pred
            elif network=='stnposecropnnunet2D':
                white = torch.ones((test_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(
                    device)
                black = torch.zeros((test_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(
                    device)
                pose = test_image.clone()
                pose = torch.where(pose >= (test_image[0].min() + 0.01),
                                   torch.where(pose < (test_image[0].min() + 0.01), pose, white), black)
                with torch.no_grad():
                    x_affine, theta = net1(torch.cat((test_image, pose), dim=1))
                    imageps = x_affine[:, :1,:,:]
                    images = list(image for image in imageps)
                    pred = net2(images)
                    theta_crop = torch.zeros((test_target.size()[0], 2, 3), requires_grad=False, dtype=torch.float).to(torch.device("cuda"))
                    for ii in range(test_image.size()[0]):
                                                if pred[ii]['boxes'].nelement() != 0 and pred[ii]['labels'][0] == 1:
                                                                        theta_crop[ii, 0, 0] = ((pred[ii]['boxes'][0][2] - pred[ii]['boxes'][0][0] + 50) / (test_target.size()[2] * 1.0))  # x2-x1/w
                                                                        theta_crop[ii, 0, 2] = ((pred[ii]['boxes'][0][2] + pred[ii]['boxes'][0][0]) / (test_target.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                                                                        theta_crop[ii, 1, 1] = ((pred[ii]['boxes'][0][3] - pred[ii]['boxes'][0][1] + 50) / (test_target.size()[3] * 1.0))  # y2-y1/h
                                                                        theta_crop[ii, 1, 2] = ((pred[ii]['boxes'][0][3] + pred[ii]['boxes'][0][1]) / (test_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
                                                else:
                                                                        theta_crop[ii, 0, 0] = 1
                                                                        theta_crop[ii, 0, 2] = 0
                                                                        theta_crop[ii, 1, 1] = 1
                                                                        theta_crop[ii, 1, 2] = 0
                    grid_cropped = F.affine_grid(theta_crop, torch.Size([imageps.size()[0],imageps.size()[1],256,256]), align_corners=False)
                    x_crop = F.grid_sample(imageps, grid_cropped, align_corners=False,
                                     mode='bilinear')  # , padding_mode="border")
                    output_crop = net3(x_crop)
                    output_affine = inv_affine(output_crop[0], theta_crop,osize=torch.Size([imageps.size()[0],imageps.size()[1],512,512]))
                    pred = inv_affine(output_affine, theta)

                if tta:
                    final_pred = (1 / 4) * pred
                    # flip x
                    with torch.no_grad():
                        output_crop = net3(th.flip(x_crop, [2]))
                        output_affine = inv_affine(th.flip(output_crop[0], [2]), theta_crop)
                        pred = inv_affine(output_affine, theta)
                    final_pred += (1 / 4) * (pred)

                    # flip y
                    with torch.no_grad():
                        output_crop = net3(th.flip(x_crop, [3]))
                        output_affine = inv_affine(th.flip(output_crop[0], [3]), theta_crop)
                        pred = inv_affine(output_affine, theta)
                    final_pred += (1 / 4) * (pred)

                    # flip x,y
                    with torch.no_grad():
                        output_crop = net3(th.flip(x_crop, [2, 3]))
                        output_affine = inv_affine(th.flip(output_crop[0], [2, 3]), theta_crop)
                        pred = inv_affine(output_affine, theta)
                    final_pred += (1 / 4) * (pred)

                    pred = final_pred   
            else:
                white = torch.ones((test_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(
                    device)
                black = torch.zeros((test_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(
                    device)
                pose = test_image.clone()
                pose = torch.where(pose >= (test_image[0].min() + 0.01),
                                   torch.where(pose < (test_image[0].min() + 0.01), pose, white), black)
                with torch.no_grad():
                    x_affine, theta = net1(torch.cat((test_image, pose), dim=1), padding='zeros')
                    x_affine[x_affine == 0] = test_image[0].min()
                    x_crop, theta_crop = net2(x_affine[:, 0, :, :].unsqueeze(1))
                    output_crop = net3(x_crop)
                    output_affine = inv_affine(output_crop[0], theta_crop)
                    pred = inv_affine(output_affine, theta)

                if tta:
                    final_pred = (1 / 4) * pred
                    # flip x
                    with torch.no_grad():
                        output_crop = net3(th.flip(x_crop, [2]))
                        output_affine = inv_affine(th.flip(output_crop[0], [2]), theta_crop)
                        pred = inv_affine(output_affine, theta)
                    final_pred += (1 / 4) * (pred)

                    # flip y
                    with torch.no_grad():
                        output_crop = net3(th.flip(x_crop, [3]))
                        output_affine = inv_affine(th.flip(output_crop[0], [3]), theta_crop)
                        pred = inv_affine(output_affine, theta)
                    final_pred += (1 / 4) * (pred)

                    # flip x,y
                    with torch.no_grad():
                        output_crop = net3(th.flip(x_crop, [2, 3]))
                        output_affine = inv_affine(th.flip(output_crop[0], [2, 3]), theta_crop)
                        pred = inv_affine(output_affine, theta)
                    final_pred += (1 / 4) * (pred)

                    pred = final_pred



            if do_seg:
                if j > 0 and g < 5:
                    if network=='nnunet2D':
                        test_target_ = ensemble(test_target,channel_dim)
                        pred_ = ensemble(pred,channel_dim)
                        img = np_to_img(test_image[0, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                        prd = np_to_img(pred_[0, 0, :, :].data.cpu().numpy(), 'target')
                        tgt = np_to_img(test_target_[0, 0, :, :].data.cpu().numpy(), 'target')
                        # 0.0.11 Store results
                        axes[g, 0].set_title("Original Test Image")
                        axes[g, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                        axes[g, 1].set_title("Predicted")
                        axes[g, 1].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                        axes[g, 2].set_title("Reference")
                        axes[g, 2].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                        g += 1
                        if g==5:
                            f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png', bbox_inches='tight')

                    elif network == 'stnpose-nnunet2D' or network == 'stncrop-nnunet2D' or network == 'stncrop3-nnunet2D' or network == 'stnposennunet2D':
                        test_target_ = ensemble(test_target,channel_dim)
                        output_affine_ = ensemble(output_affine[0],channel_dim)
                        pred_ = ensemble(pred,channel_dim)
                        img = np_to_img(test_image[0, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                        img_affine = np_to_img(x_affine[0, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                        prd_affine = np_to_img(output_affine_[0, 0, :, :].data.cpu().numpy(), 'target')
                        prd = np_to_img(pred_[0, 0, :, :].data.cpu().numpy(), 'target')
                        tgt = np_to_img(test_target_[0, 0, :, :].data.cpu().numpy(), 'target')
                        # 0.0.11 Store results
                        axes[g, 0].set_title("Original Test Image")
                        axes[g, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                        axes[g, 1].set_title("Affine Test Image")
                        axes[g, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                        axes[g, 2].set_title("Affine Predicted")
                        axes[g, 2].imshow(prd_affine, cmap='gray', vmin=0, vmax=255.)
                        axes[g, 3].set_title("Predicted")
                        axes[g, 3].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                        axes[g, 4].set_title("Reference")
                        axes[g, 4].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                        g += 1
                        if g==5:
                            f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png',
                                      bbox_inches='tight')

                    else:
                        test_target_ = ensemble(test_target,channel_dim)
                        pred_ = ensemble(pred,channel_dim)
                        output_affine_ = ensemble(output_affine,channel_dim)
                        output_crop_ = ensemble(output_crop[0],channel_dim)
                        img = np_to_img(test_image[0, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                        img_affine = np_to_img(x_affine[0, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                        img_crop = np_to_img(x_crop[0, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                        prd_crop = np_to_img(output_crop_[0, 0, :, :].data.cpu().numpy(), 'target')
                        prd_affine = np_to_img(output_affine_[0, 0, :, :].data.cpu().numpy(), 'target')
                        prd = np_to_img(pred_[0, 0, :, :].data.cpu().numpy(), 'target')
                        tgt = np_to_img(test_target_[0, 0, :, :].data.cpu().numpy(), 'target')
                        # 0.0.11 Store results
                        axes[g, 0].set_title("Original Test Image")
                        axes[g, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                        axes[g, 1].set_title("Affine Test Image")
                        axes[g, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                        axes[g, 2].set_title("Crop Test Image")
                        axes[g, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                        axes[g, 3].set_title("Crop Predicted")
                        axes[g, 3].imshow(prd_crop, cmap='gray', vmin=0, vmax=255.)
                        axes[g, 4].set_title("Affine Predicted")
                        axes[g, 4].imshow(prd_affine, cmap='gray', vmin=0, vmax=255.)
                        axes[g, 5].set_title("Predicted")
                        axes[g, 5].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                        axes[g, 6].set_title("Reference")
                        axes[g, 6].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                        g += 1
                        if g==5:
                            f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png',
                                      bbox_inches='tight')


            if ((j * batch_size) + batch_size) < n_pred:
                for m, p in zip(range(j * batch_size, j * batch_size + batch_size), range(batch_size)):
                    (d, h, w) = patch_ids[m]
                    pred_3d[:, d, h:h + patch_size[0],
                    w:w + patch_size[1]] += pred.data.cpu().numpy()[p, :]
            else:
                for m, p in zip(range(j * batch_size, n_pred), range(batch_size)):
                    (d, h, w) = patch_ids[m]
                    pred_3d[:, d, h:h + patch_size[0],
                    w:w + patch_size[1]] += pred.data.cpu().numpy()[p, :]

        print('inference DONE')
        step2 = time.time()
        if input_folder == 'children':
            dicetot, precisiontot, recalltot, hdtot = postprocessing_c(pred_3d, os.path.join(label_path, filenames_lbts[i]), patch_size, data_results,
                                       filenames_lbts[i], utils_for_post, channel_dim, 2, res)
        else:
            dicetot, precisiontot, recalltot, hdtot = postprocessing_a(pred_3d, os.path.join(label_path, filenames_lbts[i]), patch_size, data_results,
                                       filenames_lbts[i], utils_for_post, channel_dim, 2, res)
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
            mse_s3.append(msetot[2])
            hd_s3.append(hdtot[2])
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

    np.save(data_results + '/dice_test.npy', np.asarray(total_dice, dtype=np.float32))

    torch.cuda.empty_cache()

