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
from utils.utils import inv_affine, inv_affine3d, referencef3, referencef3D, referencef, keep_largest_mask_torch
from utils.figures import ensemble, np_to_img
from utils.pre_post_children import preprocessing_c,postprocessing_c, preprocessing_c25, postprocessing_c25,preprocessing_skel,postprocessing_skel, postprocessing_c_v
from utils.pre_post_adults import preprocessing_a, postprocessing_a
from Dataset import Prepare_Test_Data_new,Prepare_Test_Data_crop
from skimage.transform import resize
from skimage.morphology import skeletonize, binary_dilation, ball
from utils.losses import compute_dtm, L1, L2
from utils.pre_processing import save_nii


def test(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path,
            data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=False):
    net1 = nets[0]
    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
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

    for i in range(0,len(filenames_imts),in_c):
        step0 = time.time()
        if input_folder == 'children':
            _, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                 channel_dim, in_c, 3, res, preprocessing, do_seg, rsz)
        else:
            if channel_dim == 4:
                _, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                     channel_dim, in_c, 3, res, preprocessing, do_seg,
                                                                     input='adults')
            else:
                _, utils_for_post, x_, y_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                     channel_dim, 3, res, preprocessing, do_seg)

        step1 = time.time()

        test_loader = Prepare_Test_Data_new(patch_ids, x_, in_c, patch_size[0], patch_size[1], patch_size[2], batch_size,
                                                workers,do_seg=do_seg,segt=y_)
        net1.eval()

        print('preprocessing of', filenames_imts[i], ' DONE')

        ncols = 3  # number of columns in final grid of images
        nrows = 1  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')

        print('starting inference...')

        for j, data in enumerate(test_loader, 0):
            test_image, test_target, img = data
            test_image = test_image.to(device)
            target = test_target.to(device)
            img = img.to(device)

            with torch.no_grad():
                image_crop, theta_crop = referencef3(img, target)
                x_affine, theta = net1(test_image)

                grid_cropped = F.affine_grid(theta, img.size(), align_corners=False)
                imgg_crop = F.grid_sample(img, grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"

        loss_stn = 1 - (L1(imgg_crop, image_crop) + torch.sqrt(L2(theta, theta_crop)))

        for q in range(nrows):
            img = np_to_img(test_image[q, 0, 64, :, :].data.cpu().numpy(), 'image', massimo, minimo)
            img_crop = np_to_img(image_crop[q, 0, image_crop.size()[2]//2, :, :].data.cpu().numpy(), 'image', massimo, minimo)
            img_affine = np_to_img(imgg_crop[q, 0, imgg_crop.size()[2]//2, :, :].data.cpu().numpy(), 'image', massimo, minimo)
            # 0.0.11 Store results
            if nrows == 1:
                axes[0].set_title("Original Test Image")
                axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                axes[1].set_title("Cropped Test Image")
                axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                axes[2].set_title("Reference Crop")
                axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
            else:
                axes[q, 0].set_title("Original Test Image")
                axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                axes[q, 1].set_title("Cropped Test Image")
                axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                axes[q, 2].set_title("Reference Crop")
                axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                
        f.savefig(data_results + '/images_view1_' + str(i) + '.png', bbox_inches='tight')
        
        for q in range(nrows):
            img = np_to_img(test_image[q, 0, :, 64, :].data.cpu().numpy(), 'image', massimo, minimo)
            img_crop = np_to_img(image_crop[q, 0, :, image_crop.size()[3]//2, :].data.cpu().numpy(), 'image', massimo, minimo)
            img_affine = np_to_img(imgg_crop[q, 0, :, imgg_crop.size()[3]//2, :].data.cpu().numpy(), 'image', massimo, minimo)
            # 0.0.11 Store results
            if nrows == 1:
                axes[0].set_title("Original Test Image")
                axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                axes[1].set_title("Cropped Test Image")
                axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                axes[2].set_title("Reference Crop")
                axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
            else:
                axes[q, 0].set_title("Original Test Image")
                axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                axes[q, 1].set_title("Cropped Test Image")
                axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                axes[q, 2].set_title("Reference Crop")
                axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                
        f.savefig(data_results + '/images_view2_' + str(i) + '.png', bbox_inches='tight')
        
        for q in range(nrows):
            img = np_to_img(test_image[q, 0, :, :, 64].data.cpu().numpy(), 'image', massimo, minimo)
            img_crop = np_to_img(image_crop[q, 0, :, :, image_crop.size()[4]//2].data.cpu().numpy(), 'image', massimo, minimo)
            img_affine = np_to_img(imgg_crop[q, 0, :, :, imgg_crop.size()[4]//2].data.cpu().numpy(), 'image', massimo, minimo)
            # 0.0.11 Store results
            if nrows == 1:
                axes[0].set_title("Original Test Image")
                axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                axes[1].set_title("Cropped Test Image")
                axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                axes[2].set_title("Reference Crop")
                axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
            else:
                axes[q, 0].set_title("Original Test Image")
                axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                axes[q, 1].set_title("Cropped Test Image")
                axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                axes[q, 2].set_title("Reference Crop")
                axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                
        f.savefig(data_results + '/images_view3_' + str(i) + '.png', bbox_inches='tight')

        print('inference DONE')
        step2 = time.time()
        dices_s1.append(loss_stn)
        print('pre: ', step1 - step0)
        print('infe: ', step2 - step1)
        affine, hdr, patch_ids, x_shape, rrshape, cropshape, bbox, shape = utils_for_post
        if not os.path.exists(data_results + '/Pred'):
            os.mkdir(data_results + '/Pred')
        newpath = data_results + '/Pred/pred_' + filenames_imts[i]
        save_nii(imgg_crop[0, 0, :, :, :].data.cpu().numpy(),affine,newpath)



    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    print('dices_s1: ', dices_s1)
    total_dice = []
    s1dice = np.mean(dices_s1, axis=0)
    total_dice.append(s1dice)
    s1std = np.std(dices_s1, axis=0)
    total_dice.append(s1std)
    np.savetxt(data_results + '/s1.csv', (dices_s1), fmt='%f', delimiter=',')
    print('Structure 1: %.4f (%.4f)' % (s1dice, s1std))


    np.save(data_results + '/dice_test.npy', np.asarray(total_dice, dtype=np.float32))
    torch.cuda.empty_cache()
    
def test2D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path,
            data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=False):
    net1 = nets[0]
    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
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
    l1 = []
    l2 = []
    time_pre = []
    time_infe = []

    for i in range(0,len(filenames_imts),in_c):
        step0 = time.time()
        if input_folder == 'children':
            _, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                 channel_dim, in_c, 2, res, preprocessing, do_seg, rsz)
        else:
            if channel_dim == 4:
                _, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                     channel_dim, in_c, 2, res, preprocessing, do_seg,
                                                                     input='adults')
            else:
                _, utils_for_post, x_, y_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                     channel_dim, 2, res, preprocessing, do_seg)

        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        test_loader = Prepare_Test_Data_new(patch_ids, x_, 1, 1, patch_size[0], patch_size[1], batch_size,
                                                workers, do_seg=do_seg, segt=y_)
        net1.eval()

        print('preprocessing of', filenames_imts[i], ' DONE')
        ncols = 4  # number of columns in final grid of images
        q = 0
        nrows = 5  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')

        print('starting inference...')

        for j, data in enumerate(test_loader, 0):
            test_image, test_target = data
            test_image = test_image.squeeze(2).to(device)
            test_target = test_target.squeeze(2).to(device)

            with torch.no_grad():
                image_crop, theta_crop = referencef(test_image, ensemble(test_target, channel_dim).squeeze(1))
                x_affine, theta = net1(test_image)

            for k in range(test_image.size()[0]):
                loss_stn=L1(x_affine[k:k+1,0,:,:], image_crop[k:k+1,0,:,:])
                l1.append(loss_stn.item())
                loss_theta=L2(theta[k:k+1,:,:],theta_crop[k:k+1,:,:])
                l2.append(loss_theta.item())

            if j > 5 and q < 5:
                target = ensemble(test_target, channel_dim)
                img = np_to_img(test_image[q, 0, :, :].data.cpu().numpy(), 'image')
                img_crop = np_to_img(image_crop[q, 0, :, :].data.cpu().numpy(), 'image')
                img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image')
                img_test = np_to_img(target[q, 0, :, :].data.cpu().numpy(), 'target')

                # 0.0.11 Store results
                if nrows == 1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Cropped Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Reference Crop")
                    axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                    axes[3].set_title("Reference Mask Crop")
                    axes[3].imshow(img_test, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Cropped Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Reference Crop")
                    axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                    axes[q,3].set_title("Reference Mask Crop")
                    axes[q,3].imshow(img_test, cmap='gray', vmin=0, vmax=255.)
                q += 1
                if q == 5:
                    print('saving image')
                    f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png',
                              bbox_inches='tight')

        print('inference DONE')
        step2 = time.time()
        print('pre: ', step1 - step0)
        time_pre.append( step1 - step0)
        print('infe: ', step2 - step1)
        time_infe.append(step2 - step1)
        affine, hdr, patch_ids, x_shape, rrshape, cropshape, bbox, shape = utils_for_post
        #if not os.path.exists(data_results + '/Pred'):
        #    os.mkdir(data_results + '/Pred')
        #newpath = data_results + '/Pred/pred_' + filenames_imts[i]
        #save_nii(img_crop[0, 0, :, :, :].data.cpu().numpy(),affine,newpath)



    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    l1m = np.mean(l1, axis=0)
    l1std = np.std(l1, axis=0)
    l2m = np.mean(l2, axis=0)
    l2std = np.std(l2, axis=0)
    print('L1: %.4f (%.4f)' % (l1m, l1std))
    print('L2: %.4f (%.4f)' % (l2m, l2std))
    print('TIME PRE: %.4f (%.4f)' % (np.mean(time_pre, axis=0), np.std(time_pre, axis=0)))
    print('TIME INFE: %.4f (%.4f)' % (np.mean(time_infe, axis=0), np.std(time_infe, axis=0)))

    np.save(data_results + '/dice_test.npy', np.asarray([l1m, l1std, l2m, l2std], dtype=np.float32))

    torch.cuda.empty_cache()

def test_pose_2D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path,
            data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=False):
    net1 = nets[1]
    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
    net = nets[0]
    net.load_state_dict(torch.load(data_results + '/net_pose.pth'))
    net.eval()
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
    l1 = []
    l2 = []
    time_pre = []
    time_infe = []

    for i in range(0,len(filenames_imts),in_c):
        step0 = time.time()
        if input_folder == 'children':
            _, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                 channel_dim, in_c, 2, res, preprocessing, do_seg, rsz)
        else:
            if channel_dim == 4:
                _, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                     channel_dim, in_c, 2, res, preprocessing, do_seg,
                                                                     input='adults')
            else:
                _, utils_for_post, x_, y_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                     channel_dim, 2, res, preprocessing, do_seg)

        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        test_loader = Prepare_Test_Data_new(patch_ids, x_, 1, 1, patch_size[0], patch_size[1], batch_size,
                                                workers, do_seg=do_seg, segt=y_)
        net1.eval()

        print('preprocessing of', filenames_imts[i], ' DONE')
        ncols = 4  # number of columns in final grid of images
        q = 0
        nrows = 5  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')

        print('starting inference...')

        for j, data in enumerate(test_loader, 0):
            test_image, test_target_old = data
            test_image = test_image.squeeze(2).to(device)
            test_target_old = test_target_old.squeeze(2).to(device)
            pose = keep_largest_mask_torch(test_image)
            with torch.no_grad():
                test_imagepso, theta1 = net(torch.cat((test_image,pose), dim=1))
                test_imageps = test_imagepso[:, :1,:,:]
                grid = F.affine_grid(theta1, test_target_old.size(), align_corners=False)
                test_target = F.grid_sample(test_target_old, grid, align_corners=False, mode='nearest', padding_mode='zeros')
                image_crop, theta_crop = referencef(test_imageps, ensemble(test_target, channel_dim).squeeze(1))
                x_affine, theta = net1(test_imageps)

            for k in range(test_image.size()[0]):
                loss_stn=L1(x_affine[k:k+1,0,:,:], image_crop[k:k+1,0,:,:])
                l1.append(loss_stn.item())
                loss_theta=L2(theta[k:k+1,:,:],theta_crop[k:k+1,:,:])
                l2.append(loss_theta.item())

            if j > 5 and q < 5:
                target = ensemble(test_target, channel_dim)
                img = np_to_img(test_imageps[q, 0, :, :].data.cpu().numpy(), 'image')
                img_crop = np_to_img(image_crop[q, 0, :, :].data.cpu().numpy(), 'image')
                img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image')
                img_test = np_to_img(target[q, 0, :, :].data.cpu().numpy(), 'target')

                # 0.0.11 Store results
                if nrows == 1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Cropped Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Reference Crop")
                    axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                    axes[3].set_title("Reference Mask Crop")
                    axes[3].imshow(img_test, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Cropped Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Reference Crop")
                    axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                    axes[q,3].set_title("Reference Mask Crop")
                    axes[q,3].imshow(img_test, cmap='gray', vmin=0, vmax=255.)
                q += 1
                if q == 5:
                    print('saving image')
                    f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png',
                              bbox_inches='tight')

        print('inference DONE')
        step2 = time.time()
        print('pre: ', step1 - step0)
        time_pre.append( step1 - step0)
        print('infe: ', step2 - step1)
        time_infe.append(step2 - step1)
        affine, hdr, patch_ids, x_shape, rrshape, cropshape, bbox, shape = utils_for_post
        #if not os.path.exists(data_results + '/Pred'):
        #    os.mkdir(data_results + '/Pred')
        #newpath = data_results + '/Pred/pred_' + filenames_imts[i]
        #save_nii(img_crop[0, 0, :, :, :].data.cpu().numpy(),affine,newpath)



    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    l1m = np.mean(l1, axis=0)
    l1std = np.std(l1, axis=0)
    l2m = np.mean(l2, axis=0)
    l2std = np.std(l2, axis=0)
    print('L1: %.4f (%.4f)' % (l1m, l1std))
    print('L2: %.4f (%.4f)' % (l2m, l2std))
    print('TIME PRE: %.4f (%.4f)' % (np.mean(time_pre, axis=0), np.std(time_pre, axis=0)))
    print('TIME INFE: %.4f (%.4f)' % (np.mean(time_infe, axis=0), np.std(time_infe, axis=0)))

    np.save(data_results + '/dice_test.npy', np.asarray([l1m, l1std, l2m, l2std], dtype=np.float32))

    torch.cuda.empty_cache()


def test_faster2D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path,
                  test_data_path,
                  data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=False):
    net1 = nets[0]
    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
    print('resize: ', rsz)
    print('do seg: ', do_seg)
    print('input', in_c, ' output', channel_dim)
    print(test_data_path)
    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
    print('images:', filenames_imts)
    print('labels:', filenames_lbts)
    l1 = []
    l2 = []
    time_pre = []
    time_infe = []

    for i in range(0, len(filenames_imts), in_c):
        step0 = time.time()
        if input_folder == 'children':
            _, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                        os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                        channel_dim, in_c, 2, res, preprocessing, do_seg, rsz)
        else:
            if channel_dim == 4:
                _, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                            os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                            channel_dim, in_c, 2, res, preprocessing, do_seg,
                                                            input='adults')
            else:
                _, utils_for_post, x_, y_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                            os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                            channel_dim, 2, res, preprocessing, do_seg)

        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        test_loader = Prepare_Test_Data_new(patch_ids, x_, 1, 1, patch_size[0], patch_size[1], batch_size,
                                            workers, do_seg=do_seg, segt=y_)
        net1.eval()

        print('preprocessing of', filenames_imts[i], ' DONE')
        ncols = 4  # number of columns in final grid of images
        q = 0
        nrows = 5  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')

        print('starting inference...')

        for j, data in enumerate(test_loader, 0):
            test_image, test_target = data
            test_image = test_image.squeeze(2).to(device)
            test_target = test_target.squeeze(2).to(device)
            print(test_image.shape, test_target.shape)
            test_images = list(image for image in test_image)
            seg = ensemble(test_target, channel_dim).squeeze(1)
            boxes, labels = torch.zeros(test_target.size()[0], 4), torch.zeros(test_target.size()[0])
            preds, pred_labels = torch.zeros(test_target.size()[0], 4), torch.zeros(test_target.size()[0])
            theta_crop, theta = torch.zeros((test_target.size()[0], 2, 3), requires_grad=False, dtype=torch.float).to(
                torch.device("cuda")), torch.zeros((test_target.size()[0], 2, 3), requires_grad=False,
                                                   dtype=torch.float).to(
                torch.device("cuda"))

            for jj in range(len(test_images)):
                seg_crop = torch.nonzero(seg[jj], as_tuple=True)
                y = seg_crop[0]
                x = seg_crop[1]
                if x.nelement() == 0 or x.min() == x.max() or y.min() == y.max():
                    boxes[jj] = torch.tensor([0, 0, 512, 512]).unsqueeze(0).to(device)
                    labels[jj] = torch.zeros(1, dtype=torch.int64).to(device)
                    theta_crop[jj, 0, 0] = 1
                    theta_crop[jj, 0, 2] = 0
                    theta_crop[jj, 1, 1] = 1
                    theta_crop[jj, 1, 2] = 0
                else:
                    boxes[jj] = torch.tensor([x.min(), y.min(), x.max(), y.max()]).unsqueeze(0).to(device)
                    labels[jj] = torch.ones(1, dtype=torch.int64).to(device)

                    theta_crop[jj, 0, 0] = ((x.max() - x.min()) / (test_target.size()[2] * 1.0))  # x2-x1/w
                    theta_crop[jj, 0, 2] = ((x.max() + x.min()) / (test_target.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    theta_crop[jj, 1, 1] = ((y.max() - y.min()) / (test_target.size()[3] * 1.0))  # y2-y1/h
                    theta_crop[jj, 1, 2] = ((y.max() + y.min()) / (test_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
            print(boxes, labels, theta_crop)
            with torch.no_grad():
                pred = net1(test_images)
            for ii in range(test_image.size()[0]):
                if pred[ii]['boxes'].nelement() != 0 and pred[ii]['labels'][0] == 1:
                    preds[ii] = pred[ii]['boxes'][0]
                    pred_labels[ii] = pred[ii]['labels'][0]
                    theta[ii, 0, 0] = ((preds[ii][2] - preds[ii][0]) / (test_target.size()[2] * 1.0))  # x2-x1/w
                    theta[ii, 0, 2] = ((preds[ii][2] + preds[ii][0]) / (test_target.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    theta[ii, 1, 1] = ((preds[ii][3] - preds[ii][1]) / (test_target.size()[3] * 1.0))  # y2-y1/h
                    theta[ii, 1, 2] = ((preds[ii][3] + preds[ii][1]) / (test_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
                else:
                    theta[ii, 0, 0] = 1
                    theta[ii, 0, 2] = 0
                    theta[ii, 1, 1] = 1
                    theta[ii, 1, 2] = 0
            print(preds, pred_labels, theta)

            grid_cropped = F.affine_grid(theta_crop, test_image.size(), align_corners=False)
            image_crop = F.grid_sample(test_image, grid_cropped, align_corners=False,
                                       mode='bilinear')  # , padding_mode="border"

            grid_cropped = F.affine_grid(theta, test_image.size(), align_corners=False)
            x_affine = F.grid_sample(test_image, grid_cropped, align_corners=False,
                                     mode='bilinear')  # , padding_mode="border"

            for k in range(test_image.size()[0]):
                loss_stn = L1(x_affine[k:k + 1, 0, :, :], image_crop[k:k + 1, 0, :, :])
                l1.append(loss_stn.item())
                loss_theta = L2(theta[k:k + 1, :, :], theta_crop[k:k + 1, :, :])
                l2.append(loss_theta.item())

            if j > 5 and q < 5:
                target = ensemble(test_target, channel_dim)
                img = np_to_img(test_image[q, 0, :, :].data.cpu().numpy(), 'image')
                img_crop = np_to_img(image_crop[q, 0, :, :].data.cpu().numpy(), 'image')
                img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image')
                img_test = np_to_img(target[q, 0, :, :].data.cpu().numpy(), 'target')

                # 0.0.11 Store results
                if nrows == 1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Cropped Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Reference Crop")
                    axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                    axes[3].set_title("Reference Mask Crop")
                    axes[3].imshow(img_test, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Cropped Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Reference Crop")
                    axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 3].set_title("Reference Mask Crop")
                    axes[q, 3].imshow(img_test, cmap='gray', vmin=0, vmax=255.)
                q += 1
                if q == 5:
                    print('saving image')
                    f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png',
                              bbox_inches='tight')

        print('inference DONE')
        step2 = time.time()
        print('pre: ', step1 - step0)
        time_pre.append(step1 - step0)
        print('infe: ', step2 - step1)
        time_infe.append(step2 - step1)
        affine, hdr, patch_ids, x_shape, rrshape, cropshape, bbox, shape = utils_for_post
        # if not os.path.exists(data_results + '/Pred'):
        #    os.mkdir(data_results + '/Pred')
        # newpath = data_results + '/Pred/pred_' + filenames_imts[i]
        # save_nii(img_crop[0, 0, :, :, :].data.cpu().numpy(),affine,newpath)

    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    l1m = np.mean(l1, axis=0)
    l1std = np.std(l1, axis=0)
    l2m = np.mean(l2, axis=0)
    l2std = np.std(l2, axis=0)
    print('L1: %.4f (%.4f)' % (l1m, l1std))
    print('L2: %.4f (%.4f)' % (l2m, l2std))
    print('TIME PRE: %.4f (%.4f)' % (np.mean(time_pre, axis=0), np.std(time_pre, axis=0)))
    print('TIME INFE: %.4f (%.4f)' % (np.mean(time_infe, axis=0), np.std(time_infe, axis=0)))

    np.save(data_results + '/dice_test.npy', np.asarray([l1m, l1std, l2m, l2std], dtype=np.float32))

    torch.cuda.empty_cache()


def test_pose_faster2D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path,
                       test_data_path,
                       data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=False):
    net1 = nets[1]
    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
    net = nets[0]
    net.load_state_dict(torch.load(data_results + '/net_pose.pth'))
    net.eval()
    print('resize: ', rsz)
    print('do seg: ', do_seg)
    print('input', in_c, ' output', channel_dim)
    print(test_data_path)
    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
    print('images:', filenames_imts)
    print('labels:', filenames_lbts)
    l1 = []
    l2 = []
    time_pre = []
    time_infe = []

    for i in range(0, len(filenames_imts), in_c):
        step0 = time.time()
        if input_folder == 'children':
            _, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                        os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                        channel_dim, in_c, 2, res, preprocessing, do_seg, rsz)
        else:
            if channel_dim == 4:
                _, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                            os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                            channel_dim, in_c, 2, res, preprocessing, do_seg,
                                                            input='adults')
            else:
                _, utils_for_post, x_, y_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                            os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                            channel_dim, 2, res, preprocessing, do_seg)

        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        test_loader = Prepare_Test_Data_new(patch_ids, x_, 1, 1, patch_size[0], patch_size[1], batch_size,
                                            workers, do_seg=do_seg, segt=y_)
        net1.eval()

        print('preprocessing of', filenames_imts[i], ' DONE')
        ncols = 4  # number of columns in final grid of images
        q = 0
        nrows = 5  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')

        print('starting inference...')

        for j, data in enumerate(test_loader, 0):
            test_image, test_target_old = data
            test_image = test_image.squeeze(2).to(device)
            pose = keep_largest_mask_torch(test_image)
            test_target_old = test_target_old.squeeze(2).to(device)
            print(test_image.shape, test_target_old.shape)
            boxes, labels = torch.zeros(test_target_old.size()[0], 4), torch.zeros(test_target_old.size()[0])
            preds, pred_labels = torch.zeros(test_target_old.size()[0], 4), torch.zeros(test_target_old.size()[0])
            theta_crop, theta = torch.zeros((test_target_old.size()[0], 2, 3), requires_grad=False,
                                            dtype=torch.float).to(
                torch.device("cuda")), torch.zeros((test_target_old.size()[0], 2, 3), requires_grad=False,
                                                   dtype=torch.float).to(
                torch.device("cuda"))
            with torch.no_grad():
                test_imagepso, theta1 = net(torch.cat((test_image, pose), dim=1))
                test_imageps = test_imagepso[:, :1, :, :]
                grid = F.affine_grid(theta1, test_target_old.size(), align_corners=False)
                test_target = F.grid_sample(test_target_old, grid, align_corners=False, mode='nearest',
                                            padding_mode='zeros')
                test_images = list(image for image in test_imageps)

            seg = ensemble(test_target, channel_dim).squeeze(1)
            for jj in range(test_target.size()[0]):
                seg_crop = torch.nonzero(seg[jj], as_tuple=True)
                y = seg_crop[0]
                x = seg_crop[1]
                if x.nelement() == 0 or x.min() == x.max() or y.min() == y.max():
                    boxes[jj] = torch.tensor([0, 0, 512, 512]).unsqueeze(0).to(device)
                    labels[jj] = torch.zeros(1, dtype=torch.int64).to(device)
                    theta_crop[jj, 0, 0] = 1
                    theta_crop[jj, 0, 2] = 0
                    theta_crop[jj, 1, 1] = 1
                    theta_crop[jj, 1, 2] = 0
                else:
                    boxes[jj] = torch.tensor([x.min(), y.min(), x.max(), y.max()]).unsqueeze(0).to(device)
                    labels[jj] = torch.ones(1, dtype=torch.int64).to(device)

                    theta_crop[jj, 0, 0] = ((x.max() - x.min()) / (test_target.size()[2] * 1.0))  # x2-x1/w
                    theta_crop[jj, 0, 2] = ((x.max() + x.min()) / (test_target.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    theta_crop[jj, 1, 1] = ((y.max() - y.min()) / (test_target.size()[3] * 1.0))  # y2-y1/h
                    theta_crop[jj, 1, 2] = ((y.max() + y.min()) / (test_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
            print(boxes, labels, theta_crop)
            with torch.no_grad():
                pred = net1(test_images)
            for ii in range(test_image.size()[0]):
                if pred[ii]['boxes'].nelement() != 0 and pred[ii]['labels'][0] == 1:
                    preds[ii] = pred[ii]['boxes'][0]
                    pred_labels[ii] = pred[ii]['labels'][0]
                    theta[ii, 0, 0] = ((preds[ii][2] - preds[ii][0]) / (test_target.size()[2] * 1.0))  # x2-x1/w
                    theta[ii, 0, 2] = ((preds[ii][2] + preds[ii][0]) / (test_target.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    theta[ii, 1, 1] = ((preds[ii][3] - preds[ii][1]) / (test_target.size()[3] * 1.0))  # y2-y1/h
                    theta[ii, 1, 2] = ((preds[ii][3] + preds[ii][1]) / (test_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
                else:
                    theta[ii, 0, 0] = 1
                    theta[ii, 0, 2] = 0
                    theta[ii, 1, 1] = 1
                    theta[ii, 1, 2] = 0
            print(preds, pred_labels, theta)

            grid_cropped = F.affine_grid(theta_crop, test_imageps.size(), align_corners=False)
            image_crop = F.grid_sample(test_imageps, grid_cropped, align_corners=False,
                                       mode='bilinear')  # , padding_mode="border"

            grid_cropped = F.affine_grid(theta, test_imageps.size(), align_corners=False)
            x_affine = F.grid_sample(test_imageps, grid_cropped, align_corners=False,
                                     mode='bilinear')  # , padding_mode="border"

            for k in range(test_image.size()[0]):
                loss_stn = L1(x_affine[k:k + 1, 0, :, :], image_crop[k:k + 1, 0, :, :])
                l1.append(loss_stn.item())
                loss_theta = L2(theta[k:k + 1, :, :], theta_crop[k:k + 1, :, :])
                l2.append(loss_theta.item())

            if j > 5 and q < 5:
                target = ensemble(test_target, channel_dim)
                img = np_to_img(test_imageps[q, 0, :, :].data.cpu().numpy(), 'image')
                img_crop = np_to_img(image_crop[q, 0, :, :].data.cpu().numpy(), 'image')
                img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image')
                img_test = np_to_img(target[q, 0, :, :].data.cpu().numpy(), 'target')

                # 0.0.11 Store results
                if nrows == 1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Cropped Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Reference Crop")
                    axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                    axes[3].set_title("Reference Mask Crop")
                    axes[3].imshow(img_test, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Cropped Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Reference Crop")
                    axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 3].set_title("Reference Mask Crop")
                    axes[q, 3].imshow(img_test, cmap='gray', vmin=0, vmax=255.)
                q += 1
                if q == 5:
                    print('saving image')
                    f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png',
                              bbox_inches='tight')

        print('inference DONE')
        step2 = time.time()
        print('pre: ', step1 - step0)
        time_pre.append(step1 - step0)
        print('infe: ', step2 - step1)
        time_infe.append(step2 - step1)
        affine, hdr, patch_ids, x_shape, rrshape, cropshape, bbox, shape = utils_for_post
        # if not os.path.exists(data_results + '/Pred'):
        #    os.mkdir(data_results + '/Pred')
        # newpath = data_results + '/Pred/pred_' + filenames_imts[i]
        # save_nii(img_crop[0, 0, :, :, :].data.cpu().numpy(),affine,newpath)

    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    l1m = np.mean(l1, axis=0)
    l1std = np.std(l1, axis=0)
    l2m = np.mean(l2, axis=0)
    l2std = np.std(l2, axis=0)
    print('L1: %.4f (%.4f)' % (l1m, l1std))
    print('L2: %.4f (%.4f)' % (l2m, l2std))
    print('TIME PRE: %.4f (%.4f)' % (np.mean(time_pre, axis=0), np.std(time_pre, axis=0)))
    print('TIME INFE: %.4f (%.4f)' % (np.mean(time_infe, axis=0), np.std(time_infe, axis=0)))

    np.save(data_results + '/dice_test.npy', np.asarray([l1m, l1std, l2m, l2std], dtype=np.float32))

    torch.cuda.empty_cache()


def test_nnDet(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path,
            data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=False):
    net1 = nets[0]
    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
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
    l1 = []
    l2 = []
    time_pre = []
    time_infe = []

    for i in range(0,len(filenames_imts),in_c):
        step0 = time.time()
        if input_folder == 'children':
            _, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                 channel_dim, in_c, 3, res, preprocessing, do_seg, rsz)
        else:
            if channel_dim == 4:
                _, utils_for_post, x_, y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                     channel_dim, in_c, 3, res, preprocessing, do_seg,
                                                                     input='adults')
            else:
                _, utils_for_post, x_, y_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),
                                                                     os.path.join(label_path, filenames_lbts[i]), patch_size,
                                                                     channel_dim, 3, res, preprocessing, do_seg)

        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        test_loader = Prepare_Test_Data_crop(x_, 1, patch_size[0], patch_size[1], patch_size[2], batch_size,
                                                workers, segt=y_)
        net1.eval()

        print('preprocessing of', filenames_imts[i], ' DONE')
        ncols = 4  # number of columns in final grid of images
        q = 0
        nrows = 5  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')

        print('starting inference...')

        for j, data in enumerate(test_loader, 0):
            test_image, test_target, _ , _ = data
            test_image = test_image.to(device)
            print(test_image.shape, test_target.shape)
            print(test_image.min(), test_image.mean(), test_image.max())
            batch = {}
            batch['data'] = test_image
            test_target = test_target.unsqueeze(1).type(th.float).to(device)
            seg = test_target.squeeze(1)
            boxes, labels = torch.zeros(test_target.size()[0], 6).to("cuda"), torch.zeros(test_target.size()[0]).to(
                "cuda")
            theta_crop, theta = torch.zeros((test_target.size()[0], 3, 4), requires_grad=False, dtype=torch.float).to(
                torch.device("cuda")), torch.zeros((test_target.size()[0], 3, 4), requires_grad=False,
                                                   dtype=torch.float).to(
                torch.device("cuda"))

            for jj in range(test_image.size()[0]):
                seg_crop = torch.nonzero(seg[jj], as_tuple=True)
                y = seg_crop[0]  # depth: axis z
                x = seg_crop[1]  # row: axis y
                z = seg_crop[2]  # col: axis x
                if x.nelement() == 0 or x.min() == x.max() or y.min() == y.max() or z.min() == z.max():
                    boxes[jj] = torch.tensor([0, 0, 128, 128, 0, 128]).unsqueeze(0).to(device)
                    labels[jj] = torch.zeros(1, dtype=torch.int64).to(device)
                    theta_crop[jj, 0, 0] = 1
                    theta_crop[jj, 0, 3] = 0
                    theta_crop[jj, 1, 1] = 1
                    theta_crop[jj, 1, 3] = 0
                    theta_crop[jj, 2, 2] = 1
                    theta_crop[jj, 2, 3] = 0
                else:
                    boxes[jj] = torch.tensor([x.min(), y.min(), x.max(), y.max(), z.min(), z.max()]).unsqueeze(0).to(
                        device)
                    labels[jj] = torch.ones(1, dtype=torch.int64).to(device)

                    theta_crop[jj, 0, 0] = ((z.max() - z.min()) / (test_target.size()[3] * 1.0))  # x2-x1/w
                    theta_crop[jj, 0, 3] = ((z.max() + z.min()) / (test_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
                    theta_crop[jj, 1, 1] = ((x.max() - x.min()) / (test_target.size()[4] * 1.0))  # y2-y1/h
                    theta_crop[jj, 1, 3] = ((x.max() + x.min()) / (test_target.size()[4] * 1.0)) - 1  # x2+x1/w - 1
                    theta_crop[jj, 2, 2] = ((y.max() - y.min()) / (test_target.size()[2] * 1.0))  # z2-z1/h
                    theta_crop[jj, 2, 3] = ((y.max() + y.min()) / (test_target.size()[2] * 1.0)) - 1  # z2+z1/w - 1
            print(boxes, labels, theta_crop)
            with torch.no_grad():
                pred_labels, preds = net1.test_forward(batch)
                volume = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1]) * (preds[:, 5] - preds[:, 4])
                # print(pred['boxes'][0])
            for ii in range(test_image.size()[0]):
                if volume[ii] > 0:
                    # if pred['boxes'][ii]:
                    #    if pred['boxes'][0][ii]['box_pred_class_id']==1:
                    #        print(pred['boxes'][0][ii]['box_coords'])
                    #        preds[ii] =  torch.from_numpy(pred['boxes'][0][ii]['box_coords']).to("cuda")
                    #        pred_labels[ii] =  torch.tensor(pred['boxes'][0][ii]['box_pred_class_id']).to("cuda")
                    theta[ii, 0, 0] = ((preds[ii][2] - preds[ii][0]) / (test_target.size()[3] * 1.0))  # x2-x1/w
                    theta[ii, 0, 3] = ((preds[ii][2] + preds[ii][0]) / (test_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
                    theta[ii, 1, 1] = ((preds[ii][3] - preds[ii][1]) / (test_target.size()[4] * 1.0))  # y2-y1/h
                    theta[ii, 1, 3] = ((preds[ii][3] + preds[ii][1]) / (test_target.size()[4] * 1.0)) - 1  # x2+x1/w - 1
                    theta[ii, 2, 2] = ((preds[ii][5] - preds[ii][4]) / (test_target.size()[2] * 1.0))  # y2-y1/h
                    theta[ii, 2, 3] = ((preds[ii][5] + preds[ii][4]) / (test_target.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                else:
                    theta[ii, 0, 0] = 1
                    theta[ii, 0, 3] = 0
                    theta[ii, 1, 1] = 1
                    theta[ii, 1, 3] = 0
                    theta[ii, 2, 2] = 1
                    theta[ii, 2, 3] = 0
            print(preds, pred_labels, theta)

            grid_cropped = F.affine_grid(theta_crop, test_image.size(), align_corners=False)
            image_crop = F.grid_sample(test_image, grid_cropped, align_corners=False,
                                       mode='bilinear')  # , padding_mode="border"

            grid_cropped_x = F.affine_grid(theta, test_image.size(), align_corners=False)
            x_affine = F.grid_sample(test_image, grid_cropped_x, align_corners=False,
                                     mode='bilinear')  # , padding_mode="border"

            for k in range(test_image.size()[0]):
                loss_stn=L1(x_affine[k:k+1,0], image_crop[k:k+1,0])
                l1.append(loss_stn.item())
                loss_theta=L2(theta[k:k+1],theta_crop[k:k+1])
                l2.append(loss_theta.item())


            del test_image, test_target, seg, image_crop, x_affine

        print('inference DONE')
        step2 = time.time()
        print('pre: ', step1 - step0)
        time_pre.append( step1 - step0)
        print('infe: ', step2 - step1)
        time_infe.append(step2 - step1)
        affine, hdr, patch_ids, x_shape, rrshape, cropshape, bbox, shape = utils_for_post
        #if not os.path.exists(data_results + '/Pred'):
        #    os.mkdir(data_results + '/Pred')
        #newpath = data_results + '/Pred/pred_' + filenames_imts[i]
        #save_nii(img_crop[0, 0, :, :, :].data.cpu().numpy(),affine,newpath)



    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    l1m = np.mean(l1, axis=0)
    l1std = np.std(l1, axis=0)
    l2m = np.mean(l2, axis=0)
    l2std = np.std(l2, axis=0)
    print('L1: %.4f (%.4f)' % (l1m, l1std))
    print('L2: %.4f (%.4f)' % (l2m, l2std))
    print('TIME PRE: %.4f (%.4f)' % (np.mean(time_pre, axis=0), np.std(time_pre, axis=0)))
    print('TIME INFE: %.4f (%.4f)' % (np.mean(time_infe, axis=0), np.std(time_infe, axis=0)))

    np.save(data_results + '/dice_test.npy', np.asarray([l1m, l1std, l2m, l2std], dtype=np.float32))

    torch.cuda.empty_cache()

def test3D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=True):
    net1 = nets[0]
    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
    print('rsz',rsz)
    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
    l1 = []
    l2 = []
    time_pre = []
    time_infe = []

    for i in range(len(filenames_imts)):
        step0 = time.time()
        if input_folder == 'children':
            n_pred, utils_for_post, x_,y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]), os.path.join(label_path, filenames_lbts[i]), patch_size, channel_dim,  in_c,3, res, preprocessing, do_seg,rsz=rsz)
        else:
            n_pred, utils_for_post, x_,y_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),os.path.join(label_path, filenames_lbts[i]), patch_size, channel_dim,  in_c,3, res, preprocessing, do_seg,rsz)
        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        test_loader = Prepare_Test_Data_crop(x_, 1, patch_size[0], patch_size[1], patch_size[2], batch_size,
                                                workers,segt=y_)
        print('preprocessing of', filenames_imts[i], ' DONE')
        print('tta: ', tta)


        net1.eval()
        ncols = 5  # number of columns in final grid of images
        nrows = 3  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')
        print('starting inference...')
        for j, data in enumerate(test_loader, 0):

            test_image,test_mask,test_original,_ = data
            test_image = test_image.to(device)
            test_original = test_original.to(device)
            test_mask = test_mask.to(device)
            print(test_original.size(),test_image.size(),test_mask.size())
            image_crop, theta_crop = referencef3D(test_original, test_mask)

            with torch.no_grad():
                _, theta = net1(test_image)

                grid = F.affine_grid(theta,test_original.size(), align_corners=False)
                x_affine = F.grid_sample(test_original, grid, align_corners=False, mode='bilinear', padding_mode='border')


                grid_cropped = F.affine_grid(theta_crop, test_mask.unsqueeze(1).size(), align_corners=False)
                v_crop = F.grid_sample(test_mask.unsqueeze(1).type(th.float), grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"

            for k in range(test_image.size()[0]):
                loss_stn=L1(x_affine[k:k+1,0], image_crop[k:k+1,0])
                l1.append(loss_stn.item())
                loss_theta=L2(theta[k:k+1],theta_crop[k:k+1])
                l2.append(loss_theta.item())
            
            img0 = np_to_img(test_original[0, 0, 256, :, :].data.cpu().numpy(), 'image')
            img_i0 = np_to_img(test_image[0, 0, 64, :, :].data.cpu().numpy(), 'image')
            img_affine0 = np_to_img(x_affine[0, 0, 256, :, :].data.cpu().numpy(), 'image')
            img_mask0 = np_to_img(v_crop[0,0, 256, :, :].data.cpu().numpy(), 'target')
            ref_0 = np_to_img(image_crop[0, 0, 256, :, :].data.cpu().numpy(), 'image')

            img1 = np_to_img(test_original[0, 0, :, 256, :].data.cpu().numpy(), 'image')
            img_i1 = np_to_img(test_image[0, 0, :, 64, :].data.cpu().numpy(), 'image')
            img_affine1 = np_to_img(x_affine[0, 0, :, 256, :].data.cpu().numpy(), 'image')
            img_mask1 = np_to_img(v_crop[0,0, :, 256, :].data.cpu().numpy(), 'target')
            ref_1 = np_to_img(image_crop[0, 0, :, 256, :].data.cpu().numpy(), 'image')

            img2 = np_to_img(test_original[0, 0, :, :, 256].data.cpu().numpy(), 'image')
            img_i2 = np_to_img(test_image[0, 0, :, :, 64].data.cpu().numpy(), 'image')
            img_affine2 = np_to_img(x_affine[0, 0, :, :, 256].data.cpu().numpy(), 'image')
            img_mask2 = np_to_img(v_crop[0,0,  :, :, 256].data.cpu().numpy(), 'target')
            ref_2 = np_to_img(image_crop[0, 0, :, :, 256].data.cpu().numpy(), 'image')
            # 0.0.11 Store results
            axes[0, 0].set_title("Original Test Image")
            axes[0, 0].imshow(img0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 1].set_title("Resize Test Image")
            axes[0, 1].imshow(img_i0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 2].set_title("Crop Test Image")
            axes[0, 2].imshow(img_affine0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 3].set_title("Reference crop")
            axes[0, 3].imshow(img_mask0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 4].set_title("Reference bbox")
            axes[0, 4].imshow(ref_0, cmap='gray', vmin=0, vmax=255.)

            axes[1, 0].set_title("Original Test Image")
            axes[1, 0].imshow(img1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 1].set_title("Resize Test Image")
            axes[1, 1].imshow(img_i1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 2].set_title("Crop Test Image")
            axes[1, 2].imshow(img_affine1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 3].set_title("Reference crop")
            axes[1, 3].imshow(img_mask1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 4].set_title("Reference bbox")
            axes[1, 4].imshow(ref_1, cmap='gray', vmin=0, vmax=255.,origin='lower')

            axes[2, 0].set_title("Original Test Image")
            axes[2, 0].imshow(img2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 1].set_title("Resize Test Image")
            axes[2, 1].imshow(img_i2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 2].set_title("Crop Test Image")
            axes[2, 2].imshow(img_affine2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 3].set_title("Reference crop")
            axes[2, 3].imshow(img_mask2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 4].set_title("Reference bbox")
            axes[2, 4].imshow(ref_2, cmap='gray', vmin=0, vmax=255.,origin='lower')

            print('saving image')
            f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png',
                          bbox_inches='tight')

            del test_image, test_original, test_mask, x_affine, image_crop, v_crop, loss_stn, loss_theta

        print('inference DONE')
        step2 = time.time()

        print('pre: ', step1 - step0)
        time_pre.append( step1 - step0)
        print('infe: ', step2 - step1)
        time_infe.append(step2 - step1)

    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    l1m = np.mean(l1, axis=0)
    l1std = np.std(l1, axis=0)
    l2m = np.mean(l2, axis=0)
    l2std = np.std(l2, axis=0)
    print('L1: %.4f (%.4f)' % (l1m, l1std))
    print('L2: %.4f (%.4f)' % (l2m, l2std))
    print('TIME PRE: %.4f (%.4f)' % (np.mean(time_pre, axis=0), np.std(time_pre, axis=0)))
    print('TIME INFE: %.4f (%.4f)' % (np.mean(time_infe, axis=0), np.std(time_infe, axis=0)))

    np.save(data_results + '/dice_test.npy', np.asarray([l1m, l1std, l2m, l2std], dtype=np.float32))

    torch.cuda.empty_cache()
    
    
def test_pose3D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=True):
    net1 = nets[0]
    net1.load_state_dict(torch.load(data_results + '/net_pose.pth'))
    net1.eval()
    net2 = nets[1]
    net2.load_state_dict(torch.load(data_results + '/net_1.pth'))
    print('rsz',rsz)
    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
    l1 = []
    l2 = []
    time_pre = []
    time_infe = []
    

    for i in range(len(filenames_imts)):
        step0 = time.time()
        if input_folder == 'children':
            n_pred, utils_for_post, x_,y_ = preprocessing_c(os.path.join(test_data_path, filenames_imts[i]), os.path.join(label_path, filenames_lbts[i]), patch_size, channel_dim,  in_c,3, res, preprocessing, do_seg,rsz=rsz)
        else:
            n_pred, utils_for_post, x_,y_ = preprocessing_a(os.path.join(test_data_path, filenames_imts[i]),os.path.join(label_path, filenames_lbts[i]), patch_size, channel_dim,  in_c,3, res, preprocessing, do_seg,rsz)
        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        test_loader = Prepare_Test_Data_crop(x_, 1, patch_size[0], patch_size[1], patch_size[2], batch_size,
                                                workers,segt=y_)
        print('preprocessing of', filenames_imts[i], ' DONE')
        print('tta: ', tta)


        net2.eval()
        ncols = 6  # number of columns in final grid of images
        nrows = 3  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')
        print('starting inference...')
        for j, data in enumerate(test_loader, 0):

            test_image,test_mask,test_original,pose = data
            test_image = test_image.to(device)
            test_original = test_original.to(device)
            test_mask = test_mask.to(device)
            pose = pose.to(device)
            


            with torch.no_grad():
                imagepso,theta_x = net1(torch.cat((test_image, pose), dim=1))
                imageps = imagepso[:,:1,:,:,:]
                grid_x = F.affine_grid(theta_x,test_original.size(), align_corners=False)
                xx_affine = F.grid_sample(test_original, grid_x, align_corners=False, mode='bilinear', padding_mode='border')
                v_crop_affine = F.grid_sample(test_mask.unsqueeze(1).type(th.float), grid_x, align_corners=False, mode='bilinear')

                
                image_crop, theta_crop = referencef3D(xx_affine, v_crop_affine.squeeze(1))
                _, theta = net2(imageps)

                grid = F.affine_grid(theta,xx_affine.size(), align_corners=False)
                x_affine = F.grid_sample(xx_affine, grid, align_corners=False, mode='bilinear', padding_mode='border')

                grid_cropped = F.affine_grid(theta_crop, v_crop_affine.size(), align_corners=False)
                v_crop = F.grid_sample(v_crop_affine, grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"

            for k in range(test_image.size()[0]):
                loss_stn=L1(x_affine[k:k+1,0], image_crop[k:k+1,0])
                l1.append(loss_stn.item())
                loss_theta=L2(theta[k:k+1],theta_crop[k:k+1])
                l2.append(loss_theta.item())
            
            img0 = np_to_img(test_original[0, 0, 256, :, :].data.cpu().numpy(), 'image')
            img_x0 = np_to_img(xx_affine[0, 0, 256, :, :].data.cpu().numpy(), 'image')
            img_i0 = np_to_img(imageps[0, 0, 64, :, :].data.cpu().numpy(), 'image')
            img_affine0 = np_to_img(x_affine[0, 0, 256, :, :].data.cpu().numpy(), 'image')
            img_mask0 = np_to_img(v_crop[0,0, 256, :, :].data.cpu().numpy(), 'target')
            ref_0 = np_to_img(image_crop[0, 0, 256, :, :].data.cpu().numpy(), 'image')

            img1 = np_to_img(test_original[0, 0, :, 256, :].data.cpu().numpy(), 'image')
            img_x1 = np_to_img(xx_affine[0, 0, :, 256, :].data.cpu().numpy(), 'image')
            img_i1 = np_to_img(imageps[0, 0, :, 64, :].data.cpu().numpy(), 'image')
            img_affine1 = np_to_img(x_affine[0, 0, :, 256, :].data.cpu().numpy(), 'image')
            img_mask1 = np_to_img(v_crop[0,0, :, 256, :].data.cpu().numpy(), 'target')
            ref_1 = np_to_img(image_crop[0, 0, :, 256, :].data.cpu().numpy(), 'image')

            img2 = np_to_img(test_original[0, 0, :, :, 256].data.cpu().numpy(), 'image')
            img_x2 = np_to_img(xx_affine[0, 0, :, :, 256].data.cpu().numpy(), 'image')
            img_i2 = np_to_img(imageps[0, 0, :, :, 64].data.cpu().numpy(), 'image')
            img_affine2 = np_to_img(x_affine[0, 0, :, :, 256].data.cpu().numpy(), 'image')
            img_mask2 = np_to_img(v_crop[0,0,  :, :, 256].data.cpu().numpy(), 'target')
            ref_2 = np_to_img(image_crop[0, 0, :, :, 256].data.cpu().numpy(), 'image')
            # 0.0.11 Store results
            axes[0, 0].set_title("Original Test Image")
            axes[0, 0].imshow(img0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 1].set_title("Affine Test Image")
            axes[0, 1].imshow(img_x0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 2].set_title("Resize Test Image")
            axes[0, 2].imshow(img_i0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 3].set_title("Crop Test Image")
            axes[0, 3].imshow(img_affine0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 4].set_title("Reference crop")
            axes[0, 4].imshow(img_mask0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 5].set_title("Reference bbox")
            axes[0, 5].imshow(ref_0, cmap='gray', vmin=0, vmax=255.)

            axes[1, 0].set_title("Original Test Image")
            axes[1, 0].imshow(img1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 1].set_title("Affine Test Image")
            axes[1, 1].imshow(img_x1, cmap='gray', vmin=0, vmax=255.)
            axes[1, 2].set_title("Resize Test Image")
            axes[1, 2].imshow(img_i1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 3].set_title("Crop Test Image")
            axes[1, 3].imshow(img_affine1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 4].set_title("Reference crop")
            axes[1, 4].imshow(img_mask1, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[1, 5].set_title("Reference bbox")
            axes[1, 5].imshow(ref_1, cmap='gray', vmin=0, vmax=255.,origin='lower')

            axes[2, 0].set_title("Original Test Image")
            axes[2, 0].imshow(img2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 1].set_title("Affine Test Image")
            axes[2, 1].imshow(img_x2, cmap='gray', vmin=0, vmax=255.)
            axes[2, 2].set_title("Resize Test Image")
            axes[2, 2].imshow(img_i2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 3].set_title("Crop Test Image")
            axes[2, 3].imshow(img_affine2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 4].set_title("Reference crop")
            axes[2, 4].imshow(img_mask2, cmap='gray', vmin=0, vmax=255.,origin='lower')
            axes[2, 5].set_title("Reference bbox")
            axes[2, 5].imshow(ref_2, cmap='gray', vmin=0, vmax=255.,origin='lower')

            print('saving image')
            f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png',
                          bbox_inches='tight')

            del test_image, test_original, test_mask, x_affine, image_crop, v_crop, loss_stn, loss_theta

        print('inference DONE')
        step2 = time.time()

        print('pre: ', step1 - step0)
        time_pre.append( step1 - step0)
        print('infe: ', step2 - step1)
        time_infe.append(step2 - step1)

    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    l1m = np.mean(l1, axis=0)
    l1std = np.std(l1, axis=0)
    l2m = np.mean(l2, axis=0)
    l2std = np.std(l2, axis=0)
    print('L1: %.4f (%.4f)' % (l1m, l1std))
    print('L2: %.4f (%.4f)' % (l2m, l2std))
    print('TIME PRE: %.4f (%.4f)' % (np.mean(time_pre, axis=0), np.std(time_pre, axis=0)))
    print('TIME INFE: %.4f (%.4f)' % (np.mean(time_infe, axis=0), np.std(time_infe, axis=0)))

    np.save(data_results + '/dice_test.npy', np.asarray([l1m, l1std, l2m, l2std], dtype=np.float32))

    torch.cuda.empty_cache()


