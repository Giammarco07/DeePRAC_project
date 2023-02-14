import argparse
import os
import time
import torch as th
import torch.utils.data
import torch.cuda.amp as amp
import numpy as np
from Networks import net, net_64, net_ad, netstncrop
from models.Dataset import Prepare_Test_Data_new
from utils.pre_post_children import preprocessing_c, postprocessing_c
from utils.pre_post_adults import preprocessing_a, postprocessing_a
from utils.patches import creation_gauss_filter, creation_patch_filter
from utils.patches import gaussian_map
from utils.pre_processing import load_nii,save_nii
from utils.autocrop import stncrop
parser = argparse.ArgumentParser()

parser.add_argument("--net", required=False, default=1,
                    help="Choose between: '1', '2', '3'. e.g. '1' for kidneys and masses, '2' for ureters, and '3' for arteries and veins. Default = 1")

parser.add_argument("--prep", required=False, default=False,
                    help="Choose if do just preprocessing and save the preprocessed image or not")

parser.add_argument("--tta",  required=False, default=True,
                    help="Use if you want to use test time augmentation in the inference --> test time augmentation: possibles better results, not sure, more test time. Not suggested")

parser.add_argument("--crop",  required=False, default=False,
                    help="Auto crop")
                    
# #
print('WELCOME')
print("Use 'python main.py -h' to discover all the options.")


args = parser.parse_args()
data_input = './data'
data_results = './data'
net_type = int(args.net)
print('using net ', net_type)
test_data_path = data_input
filenames_imts = 'image.nii'
filenames_orig = 'original.nii'
tta = args.tta
if net_type>3:
    tta=False


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    t = torch.cuda.get_device_properties(0).total_memory
    #c = torch.cuda.memory_cached(0)
    #a = torch.cuda.memory_allocated(0)
    #f = c - a  # free inside cache
    batch_size = 4
    workers = 0
else:
    device = torch.device("cpu")
    batch_size = 4
    workers = 0

print('Device:', device)
print('Batchsize: ', batch_size)
print('TTA:',tta)

if args.crop:
    patch_size = np.array((128, 128, 128))
    img, affine, _ = load_nii(os.path.join(test_data_path, filenames_orig))
    net = netstncrop(1,patch_size).to(device)
    net.load_state_dict(torch.load('./models/net_stncrop.pth',map_location=device))
    preprocessing = [76.99, 67.73, 303.0, -36.0]  # mean,sd,p95,p05
    theta = stncrop(img,patch_size,preprocessing,net,device)
    scaling = (theta[:,0]).astype(int)
    print(scaling)
    translation = (theta[:,1]).astype(int)
    print(translation)
    np.savez_compressed(os.path.join(test_data_path, 'theta.npz'), scaling=scaling,translation=translation, affine=affine)
    exit()
    
patch_size_pre = np.array((96, 160, 160))
print('patch size:',patch_size_pre)
patch_size_ad = np.array((128, 128, 128))
res = [0.45703101, 0.45703101, 0.8999939]
mode = 'precision'

if net_type == 1:

    channel_dim = 3
    net = net(channel_dim).to(device)
    net.load_state_dict(torch.load('./models/net_KT_new.pth',map_location=device))
    preprocessing = [76.99, 67.73, 303.0, -36.0]  # mean,sd,p95,p05
    patch_size = patch_size_pre


elif net_type == 2:

    channel_dim = 2
    net = net(channel_dim).to(device)
    net.load_state_dict(torch.load('./models/net_U_new.pth',map_location=device))
    preprocessing = [1252.05, 904.98, 4363.0, 147.0]  # mean,sd,p95,p05
    patch_size = patch_size_pre

elif net_type == 3:

    channel_dim = 3
    net = net(channel_dim).to(device)
    net.load_state_dict(torch.load('./models/net_AV_new.pth',map_location=device))
    preprocessing = [188.93, 175.44, 1034.5, -3.0]  # mean,sd,p95,p05
    patch_size = patch_size_pre

else:
    channel_dim = 2
    net = net_ad(channel_dim).to(device)
    if net_type == 4:
        net.load_state_dict(torch.load('./models/net_B.pth',map_location=device))
    elif net_type == 5:
        net.load_state_dict(torch.load('./models/net_L.pth',map_location=device))
    else:
        net.load_state_dict(torch.load('./models/net_S.pth',map_location=device))
    preprocessing = [222.24, 314.44, 1351.0, -71.0]  # mean,sd,p95,p05
    patch_size = patch_size_ad


step0 = time.time()
if os.path.exists(os.path.join(test_data_path, 'prep.npz')) and net_type<=3 and not args.prep:
    print('Prepocessing was already done and it will be loaded')
    loaded = np.load(os.path.join(test_data_path, 'prep.npz'), allow_pickle=True)
    n_pred = loaded['n_pred0']
    x_ = loaded['x_0']
    utils_for_post = loaded['utils_for_post']
elif os.path.exists(os.path.join(test_data_path, 'prep2.npz')) and net_type>3 and not args.prep:
    print('Prepocessing was already done and it will be loaded')
    loaded = np.load(os.path.join(test_data_path, 'prep2.npz'), allow_pickle=True)
    n_pred = loaded['n_pred']
    x_ = loaded['x_']
    utils_for_post = loaded['utils_for_post']
else:
    if net_type>3:
        n_pred, utils_for_post, x_ = preprocessing_a(os.path.join(test_data_path, filenames_orig), patch_size_ad)
        np.savez_compressed(os.path.join(test_data_path, 'prep2.npz'), n_pred=n_pred, utils_for_post=utils_for_post, x_=x_)
        print('Prepocessing DONE')
    else:
        n_pred0, utils_for_post, x_0 = preprocessing_c(os.path.join(test_data_path, filenames_imts),os.path.join(test_data_path, filenames_orig), patch_size_pre)
        if args.prep:
            np.savez_compressed(os.path.join(test_data_path, 'prep.npz'), n_pred0=n_pred0, utils_for_post=utils_for_post, x_0=x_0)
            print('Prepocessing DONE')
            quit()
        else:
                n_pred = n_pred0
                x_ = x_0

step1 = time.time()
patch_ids = utils_for_post[2]
imgshape = utils_for_post[3]


pred_3d = th.as_tensor(np.zeros((channel_dim,imgshape[0],imgshape[1],imgshape[2]), dtype=float), dtype=th.float).to(device)
test_loader = Prepare_Test_Data_new(patch_ids, x_, patch_size[0],patch_size[1], patch_size[2], batch_size, workers, preprocessing)
print('preprocessing of',filenames_imts,' DONE')
print('tta: ', tta)
net.eval()
print('starting inference for ',n_pred,' patches...')

if mode=='precision':
    gaussian_importance_map = th.as_tensor(gaussian_map(patch_size), dtype=th.float).to(device)
else:
    gaussian_importance_map = th.ones(patch_size[0],patch_size[1],patch_size[2]).to(device)

for j, data in enumerate(test_loader, 0):
    test_image = data
    test_image = test_image.to(device)

    with amp.autocast():
        with torch.no_grad():
            pred, _, _, _ = net(test_image)

        if tta:
            final_pred = (1 / 8) * pred
            # flip x
            with torch.no_grad():
                pred, _, _, _ = net(th.flip(test_image, [2]))
            final_pred += (1 / 8) * (th.flip(pred, [2]))
            # flip y
            with torch.no_grad():
                pred, _, _, _ = net(th.flip(test_image, [3]))
            final_pred += (1 / 8) * (th.flip(pred, [3]))
            # flip z
            with torch.no_grad():
                pred, _, _, _ = net(th.flip(test_image, [4]))
            final_pred += (1 / 8) * (th.flip(pred, [4]))

            # flip x,y
            with torch.no_grad():
                pred, _, _, _ = net(th.flip(test_image, [2, 3]))
            final_pred += (1 / 8) * (th.flip(pred, [2, 3]))

            # flip x,z
            with torch.no_grad():
                pred, _, _, _ = net(th.flip(test_image, [2, 4]))
            final_pred += (1 / 8) * (th.flip(pred, [2, 4]))

            # flip y,z
            with torch.no_grad():
                pred, _, _, _ = net(th.flip(test_image, [3, 4]))
            final_pred += (1 / 8) * (th.flip(pred, [3, 4]))

            # flip x,y,z
            with torch.no_grad():
                pred, _, _, _ = net(th.flip(test_image, [2, 3, 4]))
            final_pred += (1 / 8) * (th.flip(pred, [2, 3, 4]))

            pred = final_pred

    # apply of gaussian filter for reconstruction
    pred[:, :] *= gaussian_importance_map

    if ((j * batch_size) + batch_size) < n_pred:
        for h,p in zip(range(j*batch_size,j*batch_size+batch_size),range(batch_size)):
            (d, h, w) = patch_ids[h]
            pred_3d[:,d:d + patch_size[0], h:h + patch_size[1], w:w + patch_size[2]] += pred[p,:]
    else:
        for h,p in zip(range(j*batch_size,n_pred),range(batch_size)):
            (d, h, w) = patch_ids[h]
            pred_3d[:,d:d + patch_size[0], h:h + patch_size[1], w:w + patch_size[2]] += pred[p,:]

if mode == 'precision':
    gauss_n = th.as_tensor(creation_gauss_filter(imgshape, patch_ids,patch_size[0],patch_size[1], patch_size[2]), dtype=th.float).to(device)
else:
    gauss_n = th.as_tensor(creation_patch_filter(imgshape, patch_ids, patch_size[0],patch_size[1], patch_size[2]), dtype=th.float).to(device)

print('normalization because of overlapping')
pred_3d /= gauss_n
pred_3d = pred_3d.data.cpu().numpy()
print('inference DONE')
step2 = time.time()
if net_type>3:
    postprocessing_a(pred_3d, data_results, filenames_imts, utils_for_post, net_type, x_)
else:
    postprocessing_c(pred_3d, data_results, filenames_imts, utils_for_post, channel_dim,net_type)
step3 = time.time()

print('pre in s: ', (step1-step0))
print('infe in s:  ', (step2-step1))
print('post in s:  ', (step3-step2))
