"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
import numpy as np
import random
import scipy.ndimage as ndimage
import skimage.measure as measure
from PIL import Image
import sys
ee = sys.float_info.epsilon

def keep_largest_mask(image_original):
    image = np.copy(image_original)
    image[image > (image.min() + 0.005)] = 1
    image[image < 1] = 0
    xx = ndimage.morphology.binary_dilation(image, iterations=2).astype(int)
    xxx = ndimage.morphology.binary_fill_holes(xx).astype(int)
    labels_mask = measure.label(xxx)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
    labels_mask[labels_mask != 0] = 1
    mask = labels_mask
    return mask


def keep_largest(image):
    mask = np.copy(image)
    mask[mask > (mask.min() + 0.05)] = 1
    mask[mask < 1] = 0
    xx = ndimage.morphology.binary_dilation(mask, iterations=1).astype(int)
    xxx = ndimage.morphology.binary_fill_holes(xx).astype(int)
    labels_mask = measure.label(xxx)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
    labels_mask[labels_mask != 0] = 1
    largest = labels_mask * image
    largest[labels_mask == 0] = image.min()
    return largest


def identify_black(patient):
    num_pixel = []
    max1,max2 = 0,0  # reset numpixel for new landmark
    landmark1 = 0 # reset position for new landmark
    landmark2 = patient.shape[2]-1 # reset position for new landmark
    for i in range(patient.shape[2]):
        slice = patient[:, :, i]
        slice = keep_largest(slice)
        mask = keep_largest_mask(slice)
        maskzero = np.copy(mask)
        maskzero[np.where(slice > (slice.min()+0.1))] = 0
        num_pixel.insert(i, np.sum(maskzero)/(np.sum(mask)+ee))
    if len(num_pixel) > 400:
        start = len(num_pixel) - 400
    else:
        start = 0
    for i in range(len(num_pixel)-1,start-1,-1):
        if i >= (len(num_pixel) - 100):  # find the max value in the least 100 slices
            if num_pixel[i] > max2:
                max2 = num_pixel[i]
                landmark2 = i
        else:
            if num_pixel[i] > max1 and (landmark2-i)>8: #8 equal to bacthsize
                max1 = num_pixel[i]
                landmark1 = i
    # print('the landmark is in position:' + str(landmark))
    return landmark1, landmark2

def identify_black_test(patient):
    num_pixel = []
    max1,max2 = 0,0  # reset numpixel for new landmark
    landmark1 = 0 # reset position for new landmark
    landmark2 = patient.shape[2] # reset position for new landmark
    for i in range(patient.shape[2]):
        slice = patient[:, :, i]
        slice = keep_largest(slice)
        mask = keep_largest_mask(slice)
        mask[np.where(slice > (slice.min()))] = 0
        num_pixel.insert(i, np.sum(mask))
    for i in range(len(num_pixel)):
        if i >= (len(num_pixel) - 100):  # find the max value in the least 100 slices
            if num_pixel[i] > max2:
                max2 = num_pixel[i]
                landmark2 = i
        else:
            if num_pixel[i] > max1:
                max1 = num_pixel[i]
                landmark1 = i
    # print('the landmark is in position:' + str(landmark))
    return landmark1, landmark2


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        if self.return_paths:
            return path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class templatedataset(data.Dataset):
    def __init__(self, dir, transform=None):
        self.transform = transform
        self.n_pat = 0
        self.stop = []
        self.lunghezze = []
        self.data_list = {}  # lista di tutti i pazienti
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname.endswith('.npz'):
                    self.data = np.load(os.path.join(root, fname), mmap_mode='r')
                    # np.insert(self.data_list, self.n_pat, self.data['arr_0'])
                    self.data_list[self.n_pat] = self.data['arr_0']
                    self.lunghezze.insert(self.n_pat, self.data['arr_0'].shape[2])  # lunghezze di ciascun paziente
                    self.stop.insert(self.n_pat, sum(self.lunghezze) - 1)
                    self.n_pat += 1
        print(self.stop[-1] + 1)
    def __len__(self):
        # length=sum(self.length_each_patient) #fa la somma complessiva del numero immagini di ciascun paziente
        length = self.stop[-1] + 1  # l'ultimo stop comporta la lunghezza totale degli elementi dell'array +1 perche si è
        # partiti da 0
        return length  # len(length)

    def __getitem__(self, idx):
        pid = 0 #identificativo paziente
        for i in range(len(self.stop)):
            if self.stop[i] >= idx:
                pid = i
                break
        if pid == 0:
            slices = idx
        else:
            slices = idx - self.stop[i - 1] - 1  # controlla l'indice se prendo l'ultima slice
        patient_data = self.data_list[pid]
        A = patient_data[:, :, slices]
        A = keep_largest(A)
        A = np.transpose(A, (1, 0))
        np.clip(A, -0.46, 4.75, out=A)
        A = (((A+0.46)/5.21)*2.0)-1.0
        A = A[np.newaxis, :, :]
        return torch.as_tensor(A, dtype=torch.float)
        
class templatedataset_test(data.Dataset):
    def __init__(self, dir, transform=None):
        self.transform = transform
        self.n_pat = 0
        self.stop = []
        self.lunghezze = []
        self.data_list = {}  # lista di tutti i pazienti
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname.endswith('.npz'):
                    self.data = np.load(os.path.join(root, fname), mmap_mode='r')
                    # np.insert(self.data_list, self.n_pat, self.data['arr_0'])
                    self.data_list[self.n_pat] = self.data['arr_0']
                    self.lunghezze.insert(self.n_pat, self.data['arr_0'].shape[2])  # lunghezze di ciascun paziente
                    self.n_pat += 1

    def __len__(self):
        length = self.n_pat
        return length

    def __getitem__(self, idx):
        patient_data = self.data_list[idx]
        slices = random.randint(0,self.lunghezze[idx])
        A = patient_data[:, :, slices]
        A = keep_largest(A)
        A = np.transpose(A, (1, 0))
        np.clip(A, -0.46, 4.75, out=A)
        A = (((A+0.46)/5.21)*2.0)-1.0
        A = A[np.newaxis, :, :]
        return torch.as_tensor(A, dtype=torch.float)
       
class templatedatasetA(data.Dataset):
    def __init__(self, dir, transform=None, batch_size=1):
        self.transforms = transform
        self.batch_size = batch_size
        self.n_pat = 0
        self.stop = []
        self.name=[]
        self.lunghezze = []
        self.data_list = {}  # list di tutti i pazienti
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname.endswith('.npz'):
                    self.data = np.load(os.path.join(root, fname), mmap_mode='r')
                    self.data_list[self.n_pat] = self.data['arr_0']
                    self.name.insert(self.n_pat, fname)
                    self.lunghezze.insert(self.n_pat, self.data['arr_0'].shape[2])  # lunghezze di ciascun paziente
                    self.stop.insert(self.n_pat, sum(self.lunghezze) - 1)
                    self.n_pat += 1

    def __len__(self):
        length = self.n_pat * (120)
        return length

    def __getitem__(self, idx):

        patient_id = int(idx / 120)
        patient_data = self.data_list[patient_id]
        patient_black, landmark = identify_black(patient_data)  # calculate the number of black pixel
        data = patient_data[:, :, landmark - 160: landmark-40]
        slice = data[:, :, (idx - patient_id * 120)]
        A = keep_largest(slice)
        A = np.transpose(A, (1, 0))
        np.clip(A, -0.46, 4.75, out=A)
        A = (((A + 0.46) / 5.21) * 2.0) - 1.0
        A = A[np.newaxis, :, :]
        return torch.as_tensor(A, dtype=torch.float)

class templatedatasetB(data.Dataset):
    def __init__(self, dir, transform=None, batch_size=1):
        self.transforms = transform
        self.batch_size = batch_size
        self.n_pat = 0
        self.stop = []
        self.name=[]
        self.lunghezze = []
        self.data_list = {}  # list di tutti i pazienti
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname.endswith('.npz'):
                    self.data = np.load(os.path.join(root, fname), mmap_mode='r')
                    self.data_list[self.n_pat] = self.data['arr_0']
                    self.name.insert(self.n_pat, fname)
                    self.lunghezze.insert(self.n_pat, self.data['arr_0'].shape[2])  # lunghezze di ciascun paziente
                    self.stop.insert(self.n_pat, sum(self.lunghezze) - 1)
                    self.n_pat += 1

    def __len__(self):
        length = self.n_pat * (140)
        return length

    def __getitem__(self, idx):

        patient_id = int(idx / 140)
        patient_data = self.data_list[patient_id]
        patient_black, landmark = identify_black(patient_data)  # calculate the number of black pixel
        data = patient_data[:, :, landmark - 190: landmark-50]
        slice = data[:, :, (idx - patient_id * 140)]
        A = keep_largest(slice)
        A = np.transpose(A, (1, 0))
        np.clip(A, -0.46, 4.75, out=A)
        A = (((A + 0.46) / 5.21) * 2.0) - 1.0
        A = A[np.newaxis, :, :]
        return torch.as_tensor(A, dtype=torch.float)

class templatedatasetA_bis(data.Dataset):
    def __init__(self, dir, zone, transform=None, batch_size=1):
        self.transforms = transform
        self.batch_size = batch_size
        self.zone = zone
        self.n_pat = 0
        self.stop = []
        self.lunghezze = []
        self.data_list = {}  # list di tutti i pazienti
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname.endswith('.npz'):
                    self.data = np.load(os.path.join(root, fname), mmap_mode='r')
                    self.data_list[self.n_pat] = self.data['arr_0']
                    self.lunghezze.insert(self.n_pat, self.data['arr_0'].shape[2])  # lunghezze di ciascun paziente
                    self.stop.insert(self.n_pat, sum(self.lunghezze) - 1)
                    self.n_pat += 1

    def __len__(self):
        if self.zone == 1:
            length = self.n_pat * 40
        if self.zone == 2:
            length = self.n_pat * 60
        if self.zone == 3:
            length = self.n_pat * 60
        if self.zone == 4:
            length = self.n_pat * 120

        return length

    def __getitem__(self, idx):

        if self.zone == 2 or self.zone == 3: # only this zone for now
            patient_id = int(idx / 60)
        patient_data = self.data_list[patient_id]
        patient_black, landmark = identify_black(patient_data)  # calculate the number of black pixel
        if self.zone == 1:
            index_random = randint(landmark - 55, landmark)  # 65 slices for zone 1 of lungs(abbasso a 55)
            slice = patient_data[:, :, index_random]
        if self.zone == 2:
            data = patient_data[:, :, landmark - 115: landmark - 55] # 60 slices for zone 2
            position = idx - (patient_id * 60)
            slice = data[:, :, position]
        if self.zone == 3:
            if landmark - 175 <= 0:
                data = patient_data[:, :, :60]
            else:
                data = patient_data[:, :, landmark - 175: landmark - 115]
            position = idx - (patient_id * 60)
            slice = data[:, :, position]
        if self.zone == 4:
            if landmark - 260 <= 0:  # nel caso in cui non ci siano almeno 260 slice nel paziente
                index_random = randint(0, landmark - 160)
            else:
                index_random = randint(landmark - 260, landmark - 160)
            slice = patient_data[:, :, index_random]

        A = keep_largest(slice)
        A = np.transpose(A, (1, 0))
        np.clip(A, -0.46, 4.75, out=A)
        A = (((A + 0.46) / 5.21) * 2.0) - 1.0
        A = A[np.newaxis, :, :]
        return torch.as_tensor(A, dtype=torch.float)
        
class templatedatasetB_bis(data.Dataset):
    def __init__(self, dir, zone, transform=None, batch_size=1):
        self.transforms = transform
        self.batch_size = batch_size
        self.zone = zone
        self.n_pat = 0
        self.stop = []
        self.name=[]
        self.lunghezze = []
        self.data_list = {}  # list di tutti i pazienti
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname.endswith('.npz'):
                    self.data = np.load(os.path.join(root, fname), mmap_mode='r')
                    self.data_list[self.n_pat] = self.data['arr_0']
                    self.name.insert(self.n_pat, fname)
                    self.lunghezze.insert(self.n_pat, self.data['arr_0'].shape[2])  # lunghezze di ciascun paziente
                    self.stop.insert(self.n_pat, sum(self.lunghezze) - 1)
                    self.n_pat += 1

    def __len__(self):
        if self.zone == 1:
            length = self.n_pat * 50
        if self.zone == 2:
            length = self.n_pat * 70
        if self.zone == 3:
            length = self.n_pat * 70
        if self.zone == 4:
            length = self.n_pat * 150
        return length

    def __getitem__(self, idx):

        if self.zone == 2 or self.zone == 3:
            patient_id = int(idx / 70)
        elif self.zone == 1:
            idx = int(idx / 50)
        elif self.zone == 4:
            idx = int(idx / 150)
        patient_data = self.data_list[patient_id]
        patient_black, landmark = identify_black(patient_data)  # calculate the number of black pixel
        if self.zone == 1:
            index_random = randint(landmark - 50, landmark)  # prima era 50
            slice = patient_data[:, :, index_random]
        if self.zone == 2:
            data = patient_data[:, :, landmark - 120: landmark - 50]  # 60 slices for zone 2
            slice = data[:, :, (idx - patient_id * 70)]

        if self.zone == 3:
            data = patient_data[:, :, landmark - 190: landmark - 120]
            slice = data[:, :, (idx - patient_id * 70)]

        if self.zone == 4:
            if landmark - 340 <= 0:  # if there are not  340 slices for the patient
                index_random = randint(0, landmark - 190)
            else:
                index_random = randint(landmark - 340, landmark - 180)  # 150 slice for colon
            slice = patient_data[:, :, index_random]

        A = keep_largest(slice)
        A = np.transpose(A, (1, 0))
        np.clip(A, -0.46, 4.75, out=A)
        A = (((A + 0.46) / 5.21) * 2.0) - 1.0
        A = A[np.newaxis, :, :]
        return torch.as_tensor(A, dtype=torch.float) 
        
class templatedatasetnew(data.Dataset):
    def __init__(self, dir, transform=None, norm=1):
        self.transform = transform
        self.n_pat = 0
        self.norm = norm
        self.data_list = {}  # lista di tutti i pazienti

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if fname.endswith('.npz'):
                    print(os.path.join(root, fname))		
                    self.data_list[self.n_pat] = os.path.join(root, fname)
                    self.n_pat += 1
        
    def __len__(self):
        length = self.n_pat
        return length

    def __getitem__(self, idx):

        print(self.data_list[idx])
        patient_data = np.load(self.data_list[idx], mmap_mode='r')
        pdata = patient_data[patient_data.files[0]]
        print(pdata.shape)
        if self.norm == 1:
            min = -0.46
            max = 4.75
        else:
            pdata = np.rollaxis(pdata[0], 0, 3)
            pdata = np.rollaxis(pdata, 1,0)
            pdata = skimage.transform.resize(pdata, (512, 512))
            print(pdata.shape)
            #data = data[:,::-1,:]
            min = -1.66
            max = 3.33
        landmark1,landmark2 = identify_black(pdata)  # calculate the number of black pixel
        data = pdata[:, :, landmark1:landmark2]
        for i in range(data.shape[2]):
            data[:,:,i] = keep_largest(data[:,:,i])
        np.clip(data, min, max, out=data)
        data = (((data - min) / (max - min)) * 2.0) - 1.0
        return torch.as_tensor(data, dtype=torch.float)
        
        
class templatedatasetnew_test(data.Dataset):
    def __init__(self, dir, transform=None, norm=1):
        self.transform = transform
        self.n_pat = 0
        self.data_list = {}  # lista di tutti i pazienti
        self.norm = norm

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname.endswith('.npz'):

                    self.data_list[self.n_pat] = os.path.join(root, fname)
                    self.n_pat += 1

    def __len__(self):
        length = self.n_pat
        return length

    def __getitem__(self, idx):


        patient_data = np.load(self.data_list[idx], mmap_mode='r')
        pdata = patient_data[patient_data.files[0]]
        if self.norm==1:
            landmark1,landmark2 = identify_black_test(pdata)  # calculate the number of black pixel
        elif self.norm == 2:
            landmark1, landmark2 = identify_black(pdata)
        else:
            pdata = pdata[0]
            pdata = np.rollaxis(pdata, 0, 3)
            pdata = np.rollaxis(pdata, 1,0)
            landmark1, landmark2 = identify_black(pdata)

        data = pdata[:, :, landmark1:landmark2]
        for i in range(data.shape[2]):
            data[:,:,i] = keep_largest(data[:,:,i])

        #data = np.rollaxis(data, 1, 0)
        if self.norm == 1:
            min = -0.46
            max = 4.75
        elif self.norm == 2:
            data = data[:,::-1,:]
            min = -1.66
            max = 3.33
        else:
            min = -1.66
            max = 2.72
            data = skimage.transform.resize(data, (512, 512))

        np.clip(data, min, max, out=data)
        data = (((data - min) / (max - min)) * 2.0) - 1.0
        print(data.min(),data.max())
        return torch.as_tensor(data, dtype=torch.float)
