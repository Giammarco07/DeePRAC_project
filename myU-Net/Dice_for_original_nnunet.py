import nibabel as nib
import numpy as np
import argparse
from os import walk
from medpy.metric.binary import hd95 as hd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils.pre_processing import seg_label_children
from skimage.morphology import skeletonize
from utils.cropping import crop_to_bbox, crop_to_nonzero
from scipy.ndimage.measurements import label
from utils.vesselness_numpy import vesselness

def remove_small_islands(image, pix):
    count = 0
    for i in range(int(image.max())+1):
        str = image == i
        c = np.count_nonzero(str)
        if c<pix:
            image[str] = 0
            count += 1
    return image, count

def load_nii(path):
    img = nib.load(path)
    hdr = img.header
    data = img.get_fdata()
    affine = img.affine
    return data, affine, hdr


def rescale(data,channels):
    rescaled = ((data - data.min()) / (data.max() - data.min())) * channels
    return np.rint(rescaled)


def dice(pred, target):
    num = pred * target
    num = np.sum(num, axis=2)
    num = np.sum(num, axis=1)
    num = np.sum(num, axis=0)

    den1 = pred
    den1 = np.sum(den1, axis=2)
    den1 = np.sum(den1, axis=1)
    den1 = np.sum(den1, axis=0)

    den2 = target
    den2 = np.sum(den2, axis=2)
    den2 = np.sum(den2, axis=1)
    den2 = np.sum(den2, axis=0)

    dice_ = (2 * num) / (den1 + den2)
    print(dice_ * 100)

    return dice_


parser = argparse.ArgumentParser()
parser.add_argument("-p", '--pred_folder', help="Must contain all modalities for each patient in the correct"
                                                " order (same as training). Files must be named "
                                                "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                "identifier (0000, 0001, etc)", required=True)
parser.add_argument('-t', "--target_folder", required=True, help="folder for saving predictions")

args = parser.parse_args()
pred_folder = args.pred_folder
target_folder = 'unet3d/nnUNet_raw_data_base/nnUNet_raw_data/' + args.target_folder + '/labelsTs'
# path='E:/DeePRAC_PROJECT/DatabaseDEEPRAC/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task200_NECKER/'
# path='E:/DeePRAC_PROJECT/DatabaseONLINE/KiTS19Challenge/Task201_KiTS19/'
# path='E:/Andrea/'
path = '/tsi/clusterhome/glabarbera/'

(_, _, filenames_target) = next(walk(path + target_folder))
(_, _, filenames_pred) = next(walk(path + pred_folder))
filenames_target = sorted(filenames_target)
filenames_pred = sorted(filenames_pred)

print(target_folder[-23:-9])

if target_folder[-23:-9]=='Task208_NECKER':
    (_, _, filenames_target_bis) = next(walk(path + target_folder + '_bis'))
    filenames_target_bis = sorted(filenames_target_bis)
    recall = np.zeros((len(filenames_target),5))


print(len(filenames_target))

dicetotk = np.zeros(len(filenames_target))
msetotk = np.zeros(len(filenames_target))
precisiontotk = np.zeros(len(filenames_target))
recalltotk = np.zeros(len(filenames_target))
hdtotk = np.zeros(len(filenames_target))
mcdtotk = np.zeros(len(filenames_target))
vsgttotk = np.zeros(len(filenames_target))
vstotk = np.zeros(len(filenames_target))

dicetott = np.zeros(len(filenames_target))
msetott = np.zeros(len(filenames_target))
precisiontott = np.zeros(len(filenames_target))
recalltott = np.zeros(len(filenames_target))
hdtott = np.zeros(len(filenames_target))
mcdtott = np.zeros(len(filenames_target))
vsgttott = np.zeros(len(filenames_target))
vstott = np.zeros(len(filenames_target))
# dicetotc = np.zeros(len(filenames_target))
# msetotc= np.zeros(len(filenames_target))
# hdtotc = np.zeros(len(filenames_target))

conf_mat_back = np.zeros((len(filenames_target),2, 2))
conf_mat_str = np.zeros((len(filenames_target),3, 3))
conf_mat = np.zeros((len(filenames_target),3, 3))
if pred_folder=='DeePRAC_Codes/TEST-207-collection/TEST-207-bones/Pred':
    	start = 5
    	end = len(filenames_target)
elif pred_folder=='DeePRAC_Codes/TEST-207-collection/TEST-207-spleen/Pred':
    	start = 0
    	end = 5	
else:
    	start = 0
    	end = len(filenames_target)

for i in range(start,end):
    pred, _, _ = load_nii(path + pred_folder + '/' + filenames_pred[i])
    print(filenames_pred[i])
    print(pred.shape)
    target, _,  hdr = load_nii(path + target_folder + '/' + filenames_target[i])
    # target = seg_label_children(labe)
    # target = rescale(target) 
    print(filenames_target[i])
    print(target.shape)
    if pred_folder=='DeePRAC_Codes/TEST-207-collection/TEST-207-bones/Pred':
    	target_n = target == 3
    	target = target_n
    elif pred_folder=='DeePRAC_Codes/TEST-207-collection/TEST-207-spleen/Pred':
    	target_n = target == 2
    	target = target_n 	
    elif pred_folder=='DeePRAC_Codes/TEST-207-collection/TEST-207-liver/Pred':
    	target_n = target == 1
    	target = target_n
    	
    target, bbox = crop_to_nonzero(target, 0)
    print(target.shape)
    pred = crop_to_bbox(pred, bbox)

    if np.sum(pred) != 0.0:
    	pred = rescale(pred,target.max())
    	#pred = rescale(pred,1.0)

    target_kidney = np.zeros(target.shape, dtype=np.int8)
    pred_kidney = np.zeros(target.shape, dtype=np.int8)
    target_tumor = np.zeros(target.shape, dtype=np.int8)
    pred_tumor = np.zeros(target.shape, dtype=np.int8)
    

    if target_folder[-23:-9] == 'Task208_NECKER':

        #structure = np.ones((3, 3, 3), dtype=np.int)
        #new = np.zeros(pred.shape, dtype=np.int8)
        #for ii in range(1,int(target.max()+1)):
        #    strr = pred == ii
        #    labeled, _ = label(strr, structure)
        #    print(np.unique(labeled))
        #    labeled_new, _ = remove_small_islands(labeled, 1000)
        #    print(np.unique(labeled))
        #    new[labeled_new>0] = ii
        #pred = new   

        target_kidney[np.where(target == 1)] = 1
        pred_kidney[np.where(pred == 1)] = 1
        #target_kidney[np.where(target == 2)] = 1
        #pred_kidney[np.where(pred == 2)] = 1
        target_tumor[np.where(target == 2)] = 1
        pred_tumor[np.where(pred == 2)] = 1

        target_bis, _, _ = load_nii(path + target_folder + '_bis' + '/' + filenames_target_bis[i])
        target_bis = crop_to_bbox(target_bis, bbox)
        print(filenames_target_bis[i])
        print(target_bis.shape)
        print(np.unique(target_bis))
        for k in range(1,6):
            t_bis = target_bis == k
            if k<4:
                recall[i,k-1] = np.sum(t_bis*pred_kidney)/np.sum(t_bis)
            else:
                recall[i,k-1] = np.sum(t_bis*pred_tumor)/np.sum(t_bis)
        print(recall[i])
    else:
        target_kidney[np.where(target == 1)] = 1
        pred_kidney[np.where(pred == 1)] = 1
        #target_kidney[np.where(target == 2)] = 1
        #pred_kidney[np.where(pred == 2)] = 1
        target_tumor[np.where(target == 2)] = 1
        pred_tumor[np.where(pred == 2)] = 1  	

    if np.sum(target_kidney) != 0.0:
        kidney_dice = dice(pred_kidney, target_kidney)
        mserrork = (np.square(target_kidney - pred_kidney)).mean(axis=None)
        precisionk = (np.sum(target_kidney * pred_kidney) / np.sum(pred_kidney))
        recallk = (np.sum(target_kidney * pred_kidney) / np.sum(target_kidney))
        #skeletonpk = skeletonize(pred_kidney)
        #skeletontk = skeletonize(target_kidney)
        if np.sum(pred_kidney) != 0.0:
            hdistancek = hd(pred_kidney, target_kidney, voxelspacing = hdr.get_zooms())
            #mcdk = hd(skeletonpk, skeletontk, voxelspacing = hdr.get_zooms())
        else:
            hdistancek = 1000
            #mcdk = 1000
        #vp = vesselness(pred_kidney)
        #vt =vesselness(target_kidney)
        #vsgtk = np.mean(np.abs(vp[target_kidney==1]-vt[target_kidney==1]))
        #print(vsgtk)
        #vsk = np.mean(np.abs(vp-vt))
        #print(vsk)
        dicetotk[i] = kidney_dice
        msetotk[i] = mserrork
        precisiontotk[i] = precisionk
        recalltotk[i] = recallk
        hdtotk[i] = hdistancek
        #mcdtotk[i] = mcdk
        #vsgttotk[i] = vsgtk
        #vstotk[i] = vsk


    if np.sum(target_tumor) != 0.0:
        tumor_dice = dice(pred_tumor, target_tumor)
        mserrort = (np.square(target_tumor - pred_tumor)).mean(axis=None)
        precisiont = (np.sum(target_tumor * pred_tumor) / np.sum(pred_tumor))
        recallt = (np.sum(target_tumor * pred_tumor) / np.sum(target_tumor))
        skeletonpt = skeletonize(pred_tumor)
        skeletontt = skeletonize(target_tumor)
        if np.sum(pred_tumor) != 0.0:
            hdistancet = hd(pred_tumor, target_tumor, voxelspacing = hdr.get_zooms())
            mcdt = hd(skeletonpt, skeletontt, voxelspacing = hdr.get_zooms())
        else:
            hdistancet = 1000
            mcdt = 1000
        vp = vesselness(pred_tumor)
        vt =vesselness(target_tumor)
        vsgtt = np.mean(np.abs(vp[target_tumor==1]-vt[target_tumor==1]))
        print(vsgtt)
        vst = np.mean(np.abs(vp-vt))
        print(vst)
        dicetott[i] = tumor_dice
        msetott[i] = mserrort
        precisiontott[i] = precisiont
        recalltott[i] = recallt        
        hdtott[i] = hdistancet
        mcdtott[i] = mcdt
        vsgttott[i] = vsgtt
        vstott[i] = vst

    if np.sum(target_kidney) != 0.0 and np.sum(target_tumor) != 0.0:
        conf_mat[i] = confusion_matrix(target.flatten(), pred.flatten())
        for j in range(3):
            conf_mat[i,j,:]/= (conf_mat[i,j,0] + conf_mat[i,j,1] + conf_mat[i,j,2]) 
	
        conf_target = target > 0
        conf_pred = pred > 0
        conf_mat_back[i] = confusion_matrix(conf_target.flatten(), conf_pred.flatten())
        conf_mat_back[i,0] /= len(conf_target[conf_target==0].flatten())
        conf_mat_back[i,1] /= len(conf_target[conf_target==1].flatten())


        str_target = target[conf_target]
        str_pred = pred[conf_target]
        conf_mat_str[i] = confusion_matrix(str_target.flatten(), str_pred.flatten()) / len(
            str_target.flatten())



print('1: media', np.mean(dicetotk) * 100, 'std dev', np.std(dicetotk) * 100)
print('2: media', np.mean(dicetott) * 100, 'std dev', np.std(dicetott) * 100)

print('dice1: ', dicetotk)
print('mse1: ', msetotk)
print('precision1: ', precisiontotk)
print('recall1: ', recalltotk)
print('hd1: ', hdtotk)
print('mcd1: ', mcdtotk)
print('vsgt1: ', vsgttotk)
print('vs1: ', vstotk)
np.savetxt(path + args.pred_folder + '/s1.csv', (dicetotk, precisiontotk, recalltotk, hdtotk, vsgttotk), fmt='%f', delimiter=',')
print('dice2: ', dicetott)
print('mse2: ', msetott)
print('precision2: ', precisiontott)
print('recall2: ', recalltott)
print('hd2: ', hdtott)
print('mcd2: ', mcdtott)
print('vsgt2: ', vsgttott)
print('vs2: ', vstott)
np.savetxt(path + args.pred_folder + '/s2.csv', (dicetott, precisiontott, recalltott, hdtott, vsgttott), fmt='%f', delimiter=',')

print('conf_matrix_back: \n ', conf_mat_back * 100)
print('conf_matrix_back: \n media \n', np.mean(conf_mat_back, axis=0) * 100, '\n std dev \n', np.std(conf_mat_back, axis=0) * 100)

print('conf_matrix_str: \n ', conf_mat_str * 100)
print('conf_matrix_str: \n media \n', np.mean(conf_mat_str, axis=0) * 100, '\n std dev \n', np.std(conf_mat_str, axis=0) * 100)

print('conf_matrix: \n ', conf_mat * 100)
print('conf_matrix: \n media \n', np.mean(conf_mat, axis=0) * 100, '\n std dev \n', np.std(conf_mat, axis=0) * 100)

if target_folder[-23:-9] == 'Task208_NECKER':
    print('recall: ', recall * 100)
    print('recall: \n media \n', np.mean(recall, axis=0) * 100,  '\n std dev \n', np.std(recall, axis=0) * 100)

'''
conf_mat = confusion_matrix(target.flatten(), pred.flatten())
plt.figure(figsize=(15, 10))
class_names = ['background', 'arteries', 'veins']
df_cm = pd.DataFrame(conf_mat, index=class_names, columns=class_names)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
plt.savefig(path +'confusion_matrix.png')
'''
