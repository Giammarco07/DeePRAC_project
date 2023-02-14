from __future__ import print_function
import torch as th
import torch.nn.parallel
import torch.utils.data
import numpy as np
import matplotlib
import pylab as plt
import os
import time
import sys
ee = sys.float_info.epsilon
from utils.utils import inv_affine
from skimage.transform import resize
from utils.utils import referencef, inv_affine, keep_largest, keep_largest_mask, referencef2, keep_largest_mask_torch
from utils.figures import ensemble, np_to_img
from utils.pre_post_children import preprocessing_c,postprocessing_c
from utils.pre_post_adults import preprocessing_a, postprocessing_a
from utils.patches import adjust_dim
from Dataset import Prepare_Test_Data_new, Prepare_Test_Data_crop
from utils.losses import L1, L2, ce, soft_dice_loss, dice_loss, dice_loss_val, soft_dice_loss_old
import torch.nn.functional as F

def test(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing, device, do_seg=True):
    net1 = nets[0]
    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))

    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
    dices = []
    l2 = []
    time_pre = []
    time_infe = []


    print('Upload reference image for pose and shape...')
    path = '/tsi/clusterhome/glabarbera/unet3d'
    ref_path = path + '/nnUNet_preprocessed/Task200_NECKER/nnUNetData_plans_v2.1_2D_stage0/Slices/NECKER_071_1601_311.npz'
    ref_img = np.load(ref_path)['arr_0'][0, :, :].astype(np.float32)
    ref_image = resize(ref_img, (patch_size), order=1, mode='constant', cval=ref_img.min(), anti_aliasing=False)
    ref_ = keep_largest(ref_image)
    ref = keep_largest_mask(ref_)

    for i in range(len(filenames_imts)):
        step0 = time.time()
        if input_folder == 'children':
            n_pred, utils_for_post, x_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                             os.path.join(label_path, filenames_lbts[i]), patch_size, channel_dim,  in_c,
                                                             2, res, preprocessing, do_seg)
        else:
            n_pred, utils_for_post, x_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                             os.path.join(label_path, filenames_lbts[i]), patch_size, channel_dim,  in_c,
                                                             2, res, preprocessing, do_seg)
        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        test_loader = Prepare_Test_Data_new(patch_ids, x_, 1, 1, patch_size[0], patch_size[1], batch_size,
                                                workers)
        print('preprocessing of', filenames_imts[i], ' DONE')
        print('tta: ', tta)


        net1.eval()
        ncols = 5  # number of columns in final grid of images

        g = 0
        nrows = 5  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')
        print('starting inference...')
        for j, data in enumerate(test_loader, 0):

            test_image,pose = data
            test_image = test_image.squeeze(2).to(device)
            pose = pose.to(device)


            #white = torch.ones((test_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
            #black = torch.zeros((test_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
            #pose = test_image.clone()
            #pose = torch.where(pose >= (test_image[0].min()),
            #                   torch.where(pose < (test_image[0].min()), pose, white), black)

            reference = th.as_tensor(ref, dtype=th.float).unsqueeze(0).repeat(test_image.size()[0], 1, 1, 1).to(
                "cuda").to(device)

            with torch.no_grad():
                x_affine, theta = net1(torch.cat((test_image, pose), dim=1))#, padding='zeros')
                #x_affine[x_affine == 0] = test_image[0].min()

            for k in range(test_image.size()[0]):
                loss_stn=dice_loss_val(x_affine[k:k+1,1,:,:],reference[k:k+1,0,:,:])
                dices.append(loss_stn.item())
                loss_l2=L2(x_affine[k:k+1,1,:,:],reference[k:k+1,0,:,:])
                l2.append(loss_l2.item())

            if j > 5 and g < 5:
                img = np_to_img(test_image[0, 0, :, :].data.cpu().numpy(), 'image', massimo, minimo)
                pos = np_to_img(pose[0, 0, :, :].data.cpu().numpy(), 'image')
                img_affine = np_to_img(x_affine[0, 0, :, :].data.cpu().numpy(), 'image', massimo, minimo)
                img_mask = np_to_img(x_affine[0, 1, :, :].data.cpu().numpy(), 'image')
                ref_ = np_to_img(ref, 'image')

                # 0.0.11 Store results
                axes[g, 0].set_title("Original Test Image")
                axes[g, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                axes[g, 1].set_title("Original Test Pose")
                axes[g, 1].imshow(pos, cmap='gray', vmin=0, vmax=255.)
                axes[g, 2].set_title("Similarity Test Image")
                axes[g, 2].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                axes[g, 3].set_title("Similarity Test Pose")
                axes[g, 3].imshow(img_mask, cmap='gray', vmin=0, vmax=255.)
                axes[g, 4].set_title("Reference Pose")
                axes[g, 4].imshow(ref_, cmap='gray', vmin=0, vmax=255.)
                g += 1
                if g==5:
                    print('saving image')
                    f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png',
                              bbox_inches='tight')

        print('inference DONE')
        step2 = time.time()

        print('pre: ', step1 - step0)
        time_pre.append( step1 - step0)
        print('infe: ', step2 - step1)
        time_infe.append(step2 - step1)

    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    dicem = np.mean(dices, axis=0)
    dicestd = np.std(dices, axis=0)
    l2m = np.mean(l2, axis=0)
    l2std = np.std(l2, axis=0)
    print('DICE: %.4f (%.4f)' % (dicem, dicestd))
    print('L2: %.4f (%.4f)' % (l2m, l2std))
    print('TIME PRE: %.4f (%.4f)' % (np.mean(time_pre, axis=0), np.std(time_pre, axis=0)))
    print('TIME INFE: %.4f (%.4f)' % (np.mean(time_infe, axis=0), np.std(time_infe, axis=0)))

    np.save(data_results + '/dice_test.npy', np.asarray([dicem, dicestd, l2m, l2std], dtype=np.float32))

    torch.cuda.empty_cache()

def test3D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=True):
    net1 = nets[0]
    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
    print('rsz',rsz)
    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
    dices = []
    l2 = []
    time_pre = []
    time_infe = []


    print('Upload reference image for pose and shape...')
    path = '/tsi/clusterhome/glabarbera/unet3d'
    ref_path = path + '/nnUNet_preprocessed/Task200_NECKER/nnUNetData_plans_v2.1_stage1/Patches2/NECKER_071_1601.npz'
    ref_im = np.load(ref_path)
    ref_img = ref_im[ref_im.files[0]][0,:, :, :].astype(np.float32)
    if ref_img.shape[0] > 512:
        d = (ref_img.shape[0] - 512) // 2
        ref_img = ref_img[d:d + 512, :, :]
    ref_img2 = adjust_dim(ref_img, 512, 512, 512)
    #ref_image = resize(ref_img, (patch_size), order=1, mode='constant', cval=ref_img.min(), anti_aliasing=False)
    #ref_ = keep_largest(ref_img)
    #ref = keep_largest_mask(ref_)
    ref_2 = keep_largest(ref_img2,mode = '3D')
    ref2 = keep_largest_mask(ref_2,mode = '3D')

    for i in range(len(filenames_imts)):
        step0 = time.time()
        if input_folder == 'children':
            n_pred, utils_for_post, x_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                             os.path.join(label_path, filenames_lbts[i]), patch_size, channel_dim,  in_c,
                                                             3, res, preprocessing, do_seg,rsz=rsz)
        else:
            n_pred, utils_for_post, x_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                             os.path.join(label_path, filenames_lbts[i]), patch_size, channel_dim,  in_c,
                                                             3, res, preprocessing, do_seg,rsz)
        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        test_loader = Prepare_Test_Data_crop(x_, 1, patch_size[0], patch_size[1], patch_size[2], batch_size,
                                                workers)
        print('preprocessing of', filenames_imts[i], ' DONE')
        print('tta: ', tta)


        net1.eval()
        ncols = 6  # number of columns in final grid of images
        nrows = 3  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')
        print('starting inference...')
        for j, data in enumerate(test_loader, 0):

            test_image,_,test_original,pose = data
            test_image = test_image.to(device)
            test_original = test_original.to(device)
            pose = pose.to(device)

            #white = torch.ones((test_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
            #black = torch.zeros((test_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
            #pose = test_image.clone()
            #pose = torch.where(pose >= (test_image[0].min()),
            #                   torch.where(pose < (test_image[0].min()), pose, white), black)

            reference = th.as_tensor(ref2, dtype=th.float).unsqueeze(0).repeat(test_image.size()[0], 1, 1, 1, 1).to(
                "cuda").to(device)

            with torch.no_grad():
                _, theta = net1(torch.cat((test_image, pose), dim=1))
                grid = F.affine_grid(theta,test_original.size(), align_corners=False)
                x_affine = F.grid_sample(test_original, grid, align_corners=False, mode='bilinear', padding_mode='border')
                pose_x = keep_largest_mask(x_affine[0,0].data.cpu().numpy(),mode='3D')
                pose_x = torch.from_numpy(pose_x).type('torch.FloatTensor').to(device)

                loss_stn=dice_loss_val(pose_x,reference[0,0])
                dices.append(loss_stn.item())
                loss_l2=L2(pose_x,reference[0,0])
                l2.append(loss_l2.item())

            img0 = np_to_img(test_original[0, 0, 256, :, :].data.cpu().numpy(), 'image')
            img_i0 = np_to_img(test_image[0, 0, 64, :, :].data.cpu().numpy(), 'image')
            img_pose0 = np_to_img(pose[0, 0, 64, :, :].data.cpu().numpy(), 'image')
            img_affine0 = np_to_img(x_affine[0, 0, 256, :, :].data.cpu().numpy(), 'image')
            img_mask0 = np_to_img(pose_x[256, :, :].data.cpu().numpy(), 'image')
            ref_0 = np_to_img(reference[0, 0, 256, :, :].data.cpu().numpy(), 'image')

            img1 = np_to_img(test_original[0, 0, :, 256, :].data.cpu().numpy(), 'image')
            img_i1 = np_to_img(test_image[0, 0, :, 64, :].data.cpu().numpy(), 'image')
            img_pose1 = np_to_img(pose[0, 0, :, 64, :].data.cpu().numpy(), 'image')
            img_affine1 = np_to_img(x_affine[0, 0, :, 256, :].data.cpu().numpy(), 'image')
            img_mask1 = np_to_img(pose_x[:, 256, :].data.cpu().numpy(), 'image')
            ref_1 = np_to_img(reference[0, 0, :, 256, :].data.cpu().numpy(), 'image')

            img2 = np_to_img(test_original[0, 0, :, :, 256].data.cpu().numpy(), 'image')
            img_i2 = np_to_img(test_image[0, 0, :, :, 64].data.cpu().numpy(), 'image')
            img_pose2 = np_to_img(pose[0, 0, :, :, 64].data.cpu().numpy(), 'image')
            img_affine2 = np_to_img(x_affine[0, 0, :, :, 256].data.cpu().numpy(), 'image')
            img_mask2 = np_to_img(pose_x[ :, :, 256].data.cpu().numpy(), 'image')
            ref_2 = np_to_img(reference[0, 0, :, :, 256].data.cpu().numpy(), 'image')
            # 0.0.11 Store results
            axes[0, 0].set_title("Original Test Image")
            axes[0, 0].imshow(img0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 1].set_title("Resize Test Image")
            axes[0, 1].imshow(img_i0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 2].set_title("Resize Test Pose")
            axes[0, 2].imshow(img_pose0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 3].set_title("Similarity Test Image")
            axes[0, 3].imshow(img_affine0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 4].set_title("Similarity Test Pose")
            axes[0, 4].imshow(img_mask0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 5].set_title("Reference Pose")
            axes[0, 5].imshow(ref_0, cmap='gray', vmin=0, vmax=255.)

            axes[1, 0].set_title("Original Test Image")
            axes[1, 0].imshow(img1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 1].set_title("Resize Test Image")
            axes[1, 1].imshow(img_i1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 2].set_title("Resize Test Pose")
            axes[1, 2].imshow(img_pose1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 3].set_title("Similarity Test Image")
            axes[1, 3].imshow(img_affine1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 4].set_title("Similarity Test Pose")
            axes[1, 4].imshow(img_mask1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 5].set_title("Reference Pose")
            axes[1, 5].imshow(ref_1, cmap='gray', vmin=0, vmax=255.,origin='lower')

            axes[2, 0].set_title("Original Test Image")
            axes[2, 0].imshow(img2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 1].set_title("Resize Test Image")
            axes[2, 1].imshow(img_i2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 2].set_title("Resize Test Pose")
            axes[2, 2].imshow(img_pose2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 3].set_title("Similarity Test Image")
            axes[2, 3].imshow(img_affine2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 4].set_title("Similarity Test Pose")
            axes[2, 4].imshow(img_mask2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 5].set_title("Reference Pose")
            axes[2, 5].imshow(ref_2, cmap='gray', vmin=0, vmax=255.,origin='lower')

            print('saving image')
            f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png',
                          bbox_inches='tight')

            del test_image, test_original, pose, x_affine, pose_x, reference, loss_stn, loss_l2

        print('inference DONE')
        step2 = time.time()

        print('pre: ', step1 - step0)
        time_pre.append( step1 - step0)
        print('infe: ', step2 - step1)
        time_infe.append(step2 - step1)

    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    dicem = np.mean(dices, axis=0)
    dicestd = np.std(dices, axis=0)
    l2m = np.mean(l2, axis=0)
    l2std = np.std(l2, axis=0)
    print('DICE: %.4f (%.4f)' % (dicem, dicestd))
    print('L2: %.4f (%.4f)' % (l2m, l2std))
    print('TIME PRE: %.4f (%.4f)' % (np.mean(time_pre, axis=0), np.std(time_pre, axis=0)))
    print('TIME INFE: %.4f (%.4f)' % (np.mean(time_infe, axis=0), np.std(time_infe, axis=0)))

    np.save(data_results + '/dice_test.npy', np.asarray([dicem, dicestd, l2m, l2std], dtype=np.float32))

    torch.cuda.empty_cache()
