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
from utils.pre_post_children import preprocessing_c_p, postprocessing_c_p
from Dataset import Prepare_Test_Data_new


def test(input_folder, patch_size, batch_size, workers, network, nets, label_path, test_data_path,
             data_results, massimo, minimo, tta, res, preprocessing, device, rsz):
    net = nets[0]
    if os.path.exists(data_results + '/pnet_3d.pth'):
        net.load_state_dict(torch.load(data_results + '/pnet_3d.pth'))
    print('resize: ', rsz)
    print(test_data_path)
    (_, _, filenames_imts) = next(os.walk(test_data_path))
    (_, _, filenames_lbts) = next(os.walk(label_path))
    filenames_imts = sorted(filenames_imts)
    filenames_lbts = sorted(filenames_lbts)
    dices = []

    for i in range(len(filenames_imts)):
        step0 = time.time()
        n_pred, utils_for_post, x_, y_ = preprocessing_c_p(os.path.join(test_data_path, filenames_imts[i]),
                                                             os.path.join(label_path, filenames_lbts[i]),
                                                             3, 3, res, preprocessing)

        step1 = time.time()
        patch_ids = utils_for_post[2]
        imgshape = utils_for_post[3]
        pred_3d = np.ones((imgshape[0], imgshape[1], imgshape[2]), dtype=np.float32)
        pred_unet3d = np.zeros((imgshape[0], imgshape[1], imgshape[2]), dtype=np.float32)
        target_3d = np.ones((imgshape[0], imgshape[1], imgshape[2]), dtype=np.float32)
        target_unet3d = np.zeros((imgshape[0], imgshape[1], imgshape[2]), dtype=np.float32)
        test_loader = Prepare_Test_Data_new(patch_ids, x_, patch_size[0], patch_size[1], patch_size[2],
                                            batch_size,
                                            workers, do_seg=True, rsz=False, segt=y_)

        net.eval()

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
        tp, tn, fp, fn = 0, 0, 0, 0
        for j, data in enumerate(test_loader, 0):

            test_image, test_target = data
            test_image = test_image.to(device)
            test_target = test_target.to(device)
            test_seg = torch.argmax(test_target, dim=1)
            batch = test_image.size()[0]
            test_target_patches = torch.zeros((batch, 1), dtype=torch.float).to(device)
            for bb in range(batch):
                if torch.is_nonzero(torch.sum(test_seg[bb])):
                    test_target_patches[bb] = 1

            import torch.cuda.amp as amp

            if network == 'pnet3D':
                with amp.autocast():
                    with torch.no_grad():
                        pred = torch.sigmoid(net(test_image))
                        pred[pred < 0.5] = 0
                        pred[pred >= 0.5] = 1

                if ((j * batch_size) + batch_size) < n_pred:
                    for h, p in zip(range(j * batch_size, j * batch_size + batch_size), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        pp = pred.data.cpu().numpy()[p]
                        ttpp = test_target_patches.data.cpu().numpy()[p]
                        if pp == 0 and ttpp == 0:
                            tn += 1
                        if pp == 1 and ttpp == 1:
                            tp += 1
                        if pp == 1 and ttpp == 0:
                            fp += 1
                        if pp == 0 and ttpp == 1:
                            fn += 1
                        pred_3d[d:d + patch_size[0], h:h + patch_size[1],
                        w:w + patch_size[2]] -= (1 - pp)
                        pred_unet3d[d:d + patch_size[0], h:h + patch_size[1],
                        w:w + patch_size[2]] += pp
                        target_3d[d:d + patch_size[0], h:h + patch_size[1],
                        w:w + patch_size[2]] -= (1-ttpp)
                        target_unet3d[d:d + patch_size[0], h:h + patch_size[1],
                        w:w + patch_size[2]] += ttpp
                else:
                    for h, p in zip(range(j * batch_size, n_pred), range(batch_size)):
                        (d, h, w) = patch_ids[h]
                        pp = pred.data.cpu().numpy()[p]
                        ttpp = test_target_patches.data.cpu().numpy()[p]
                        if pp == 0 and ttpp == 0:
                            tn += 1
                        if pp == 1 and ttpp == 1:
                            tp += 1
                        if pp == 1 and ttpp == 0:
                            fp += 1
                        if pp == 0 and ttpp == 1:
                            fn += 1
                        pred_3d[d:d + patch_size[0], h:h + patch_size[1],
                        w:w + patch_size[2]] -= (1 - pp)
                        pred_unet3d[d:d + patch_size[0], h:h + patch_size[1],
                        w:w + patch_size[2]] += pp
                        target_3d[d:d + patch_size[0], h:h + patch_size[1],
                        w:w + patch_size[2]] -= (1-ttpp)
                        target_unet3d[d:d + patch_size[0], h:h + patch_size[1],
                        w:w + patch_size[2]] += ttpp


        postprocessing_c_p(pred_3d,data_results, 'pred_'+filenames_lbts[i], utils_for_post, 3, res)
        postprocessing_c_p(pred_unet3d, data_results, 'prd_' + filenames_lbts[i], utils_for_post, 3, res)
        postprocessing_c_p(target_3d,data_results, 'target_'+filenames_lbts[i], utils_for_post, 3, res)
        postprocessing_c_p(target_unet3d, data_results, 'tgt_'+filenames_lbts[i], utils_for_post, 3, res)
        print('tp:', tp)
        print('tn:', tn)
        print('fp:', fp)
        print('fn:', fn)
        p = tp / (tp + fp + ee)
        r = tp / (tp + fn + ee)
        f1_score = 2 * (p * r) / (p + r + ee)
        print('precision, recall, f1_score : ',p,r,f1_score)
        print('inference DONE')
        step2 = time.time()

        dices.append(f1_score)
        print('pre: ', step1 - step0)
        print('infe: ', step2 - step1)

    print('TEST TOTAL AFTER 3D RECONSTRUCTION')
    print('f1_score: ', dices)
    total_dice = []
    s1dice = np.mean(dices, axis=0)
    total_dice.append(s1dice)
    s1std = np.std(dices, axis=0)
    total_dice.append(s1std)
    print('Structure 1: %.4f (%.4f)' % (s1dice, s1std))

    np.save(data_results + '/dice_test.npy', np.asarray(total_dice, dtype=np.float32))

    torch.cuda.empty_cache()

