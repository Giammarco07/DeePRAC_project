from __future__ import print_function
import torch as th
import torch.nn.parallel
import torch.utils.data
import numpy as np
from utils.patches import gaussian_map
import matplotlib
matplotlib.use('agg')
import pylab as plt
import os
import time
import sys
ee = sys.float_info.epsilon
from utils.figures import np_to_img
from utils.pre_post_children import preprocessing_c25, postprocessing_c25
from skimage.morphology import skeletonize

def test_25d(nets, channel_dim, label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing, device):
    net = nets[0]
    net.load_state_dict(torch.load(data_results + '/net.pth'))

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


    gaussian_importance_map = th.as_tensor(gaussian_map((64,64)), dtype=th.float).to(device)


    for i in range(len(filenames_imts)):
        step0 = time.time()
        img,seg,segorig, utils_for_post = preprocessing_c25(os.path.join(test_data_path, filenames_imts[i]),
                                                                 os.path.join(label_path, filenames_lbts[i]),
                                                                 channel_dim, res, preprocessing)

        step1 = time.time()
        imgshape = img.shape
        pred_3d = np.zeros((channel_dim, imgshape[0], imgshape[1], imgshape[2]), dtype=np.float32)
        ncols = 3  # number of columns in final grid of images
        g = 0
        nrows = 5  # looking at all images takes some time

        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')
        print('starting inference...')
        for k in range(1, img.shape[0] - 2):
            skeleton = skeletonize(seg[k, :, :])
            coordinates_grid = np.ones((2, skeleton.shape[0], skeleton.shape[1]), dtype=np.int16)
            coordinates_grid[0] = coordinates_grid[0] * np.array([range(skeleton.shape[0])]).T
            coordinates_grid[1] = coordinates_grid[1] * np.array([range(skeleton.shape[0])])

            mask = skeleton != 0
            non_zero_coords = np.hstack((coordinates_grid[0][mask].reshape(-1, 1),
                                         coordinates_grid[1][mask].reshape(-1, 1)))
            for j in range(len(non_zero_coords)):
                loc = non_zero_coords[j]
                patches = np.asarray(img[k - 1:k + 2, loc[0] - 32:loc[0] + 32, loc[1] - 32:loc[1] + 32],
                                     dtype=np.float32)
                patcheseg = np.asarray(segorig[:,k - 1:k + 2, loc[0] - 32:loc[0] + 32, loc[1] - 32:loc[1] + 32],
                                     dtype=np.float32)
                test_image = torch.from_numpy(patches).type(torch.bfloat16)
                test_target = torch.from_numpy(patcheseg).type(torch.long)
                test_image = test_image.unsqueeze(0).to(device)
                test_target = test_target.unsqueeze(0).to(device)

                import torch.cuda.amp as amp

                with amp.autocast():
                    with torch.no_grad():
                        pred = net(test_image)

                    if tta:
                        final_pred = (1 / 4) * pred
                        # flip x
                        with torch.no_grad():
                            pred = net(th.flip(test_image, [2]))
                        final_pred += (1 / 4) * (th.flip(pred, [2]))
                        # flip y
                        with torch.no_grad():
                            pred = net(th.flip(test_image, [3]))
                        final_pred += (1 / 4) * (th.flip(pred, [3]))

                        # flip x,y
                        with torch.no_grad():
                            pred = net(th.flip(test_image, [2, 3]))
                        final_pred += (1 / 4) * (th.flip(pred, [2, 3]))

                        pred = final_pred

                pred[:, :] *= gaussian_importance_map

                if g < 5:
                    pred_ = torch.argmax(pred,dim=1, keepdim=True)
                    test_target_ = torch.argmax(test_target, dim=1, keepdim=True)
                    img_ = np_to_img(test_image[0, 1, :, :].type(torch.float16).data.cpu().numpy(), 'image',massimo, minimo)
                    prd_ = np_to_img(pred_[0, 0, :, :].data.cpu().numpy(), 'target')
                    tgt_ = np_to_img(test_target_[0, 0,1, :, :].data.cpu().numpy(), 'target')
                    # 0.0.11 Store results
                    axes[g, 0].set_title("Original Test Image")
                    axes[g, 0].imshow(img_, cmap='gray', vmin=0, vmax=255.)
                    axes[g, 1].set_title("Predicted")
                    axes[g, 1].imshow(prd_, cmap='gray', vmin=0, vmax=255.)
                    axes[g, 2].set_title("Reference")
                    axes[g, 2].imshow(tgt_, cmap='gray', vmin=0, vmax=255.)
                    g += 1
                    if g==5:
                        f.savefig(data_results + '/images_test_' + filenames_imts[i][-20:-11] + '.png', bbox_inches='tight')

                pred_3d[:, k, loc[0] - 32:loc[0] + 32, loc[1] - 32:loc[1] + 32] += pred.data.cpu().numpy()[0, :]

        print('inference DONE')
        step2 = time.time()
        dicetot, msetot, hdtot = postprocessing_c25(pred_3d, os.path.join(label_path, filenames_lbts[i]), data_results,
                                       filenames_lbts[i], utils_for_post, channel_dim, res)
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

    np.save(data_results + '/dice_test.npy', np.asarray(total_dice, dtype=np.float32))

    torch.cuda.empty_cache()