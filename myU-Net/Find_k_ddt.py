from scipy.ndimage import distance_transform_edt as distance
import numpy as np
from os import walk
from utils.pre_processing import load_nii
def compute_ddt(img_gt, maxx):

    for c in range(img_gt.shape[0]):
            posmask = img_gt[c].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                m = np.rint(posdis).max()
                del posmask
                print(c,':',m)
                if m>maxx:
                   maxx = m
    return maxx

path = '/tsi/clusterhome/glabarbera/unet3d/nnUNet_preprocessed/Task208_NECKER/nnUNetData_plans_v2.1_stage1/'
pred_folder = 'Patches2'
(_, _, filenames) = next(walk(path + pred_folder))
maxx = 0
filenames = sorted(filenames)
for i in range(len(filenames)):
    #pred, _, _ = load_nii(path + pred_folder + '/' + filenames[i])
    pred = np.load(path + pred_folder + '/' + filenames[i])['arr_0'][1, :, :, :]
    print(filenames[i])
    target = np.zeros((3,pred.shape[0],pred.shape[1],pred.shape[2]))
    for j in range(3):
        s = np.zeros(pred.shape).astype(np.uint8)
        s[np.where(pred == j)] = 1
        target[j] = s
    del pred
    maxx = compute_ddt(target[1:], maxx)
    
print(maxx)

