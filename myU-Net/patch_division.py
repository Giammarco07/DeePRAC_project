import numpy as np
import os
from scipy.ndimage import distance_transform_edt as distance
from utils.vesselness_numpy import vesselness

#Patches check for patch with structures
#Orientation: z,x,y
#Diameter: dm<12,dm>=12
#Hetereogeneity: std<50, std>=50
groups = np.zeros((3,3,3))

task = '208'
path = '/home/infres/glabarbera/nnunet/nnUNet_preprocessed/Task'+task+'_NECKER/nnUNetData_plans_v2.1_stage1'
img_folder = '/Patches64_eigenvalues'

(_, _, filenames_img) = next(os.walk(path + img_folder))
filenames_img = sorted(filenames_img)

folder = path + '/Patches64_oriented'
if not os.path.exists(folder):
    os.mkdir(folder)

dc,oc,nl = 0,0,0

for i in range(len(filenames_img)):
    images = np.load(path + img_folder + '/' + filenames_img[i])
    img = images[images.files[0]][0, :, :, :].astype(np.float32)
    seg = images[images.files[0]][-1, :, :, :].astype(np.long)
    img = (img*175.44)+188.93
    seg[seg>0]=1
    if np.sum(seg) != 0:
        eigen = vesselness(img*seg)
        eig = np.argmax(eigen[seg==1], axis=1)
        counts = np.bincount(eig)
        ort = np.argmax(counts)      
        print(ort)
        posdis = distance(seg)
        dmt = np.rint(posdis).max()
        hti = np.std(img[seg==1])
        hte = np.abs(np.mean(img[seg==1]) - 175.44)

        groups[ort, int(dmt >= 3) + int(dmt >= 5), int(hte >= 50) + (int(hti >= 50) & int(hte >= 50))] += 1

        if ort>0:
            print('X or Y')
            np.savez_compressed(folder + '/' + filenames_img[i], images[images.files[0]])
            oc += 1
        else:
            dc +=1
    else:
        print('no elements')
        nl +=1
print('# Patches with structures oriented in z: ',dc)
print('# Patches with structures oriented in x or y: ',oc)
print('# Patches with no structures: ',nl)
print('tensore gruppi:',groups)














