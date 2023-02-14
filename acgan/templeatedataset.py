import os
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
ee = sys.float_info.epsilon
import scipy.ndimage as ndimage
import skimage.measure as measure
import skimage.transform
import nibabel as nib


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

def keep_largest_mask(image_original):
    image = np.copy(image_original)
    image[image > (image.min() + 0.005)] = 1
    image[image < 1] = 0
    xx = ndimage.morphology.binary_dilation(image, iterations=2).astype(int)  # default int=2
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

class templatedataset_nii(Dataset):
    def __init__(self, dir, transform=None):
        self.transform = transform
        self.n_pat = 0
        self.stop = []
        self.name_list = {}
        self.data_list = {}  # lista di tutti i pazienti

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname.endswith('.nii.gz'):

                    self.data_list[self.n_pat] = os.path.join(root, fname)
                    self.name_list[self.n_pat] = fname
                    self.n_pat += 1


    def __len__(self):
        length = self.n_pat
        return length 

    def __getitem__(self, idx):

        patient_data = nib.load(self.data_list[idx])
        A = patient_data.get_fdata()
        name = self.name_list[idx]
        print(name[:15])
	
        for i in range(A.shape[2]):
            A[:,:,i] = keep_largest(A[:,:,i])

        minn = -70
        maxx = 303
        mean = -39
        sd = 70


        np.clip(A, minn, maxx, out=A)
        A = (A-mean)/sd

        min = -0.46
        max = 4.75
        np.clip(A, min, max, out=A)
        A = (((A - min) / (max - min)) * 2.0) - 1.0


        A = A[np.newaxis, :, :,:]
        return torch.as_tensor(A, dtype=torch.float), name[:15], min,max, self.data_list[idx]

class templatedataset(Dataset):
    def __init__(self, dir, transform=None):
        self.transform = transform
        self.n_pat = 0
        self.stop = []
        self.lunghezze = []
        self.data_list = {}  # lista di tutti i pazienti
        self.seg_list = {}  # lista di tutti i pazienti

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
        A = patient_data['data'][0, :, :, :]
        seg = patient_data['data'][1, :, :, :]

        A = np.rollaxis(A, 0, 3)
        shape = A.shape[:2]
        A = skimage.transform.resize(A, (512, 512))

        for i in range(A.shape[2]):
            A[:,:,i] = keep_largest(A[:,:,i])


        #minimum = (-12.0 - 175.91)/100.48
        #maximum = (631.98 - 175.91)/100.48
        #minimum = (-98.0 - 36.03)/43.69
        #maximum = (112.0 - 36.03)/43.69
        minimum = (-3.0 - 188.93)/175.44 #208
        maximum = (1034.5 - 188.93)/175.44
        #minimum = (-36.0 - 76.99)/67.73 #200
        #maximum = (303.0 - 76.99)/67.73
        #minimum = (-147.0 - 1252.05)/904.98 #202
        #maximum = (4363.0- 1252.05)/904.98

        np.clip(A, minimum, maximum, out=A)
        A = (((A - minimum) / (maximum-minimum)) * 2.0) - 1.0


        A = A[np.newaxis, :, :,:]
        return torch.as_tensor(A, dtype=torch.float), seg, self.data_list[idx], torch.as_tensor(shape, dtype=torch.float),minimum,maximum

class templatedatasetnew(Dataset):
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

class templatedatasetnew_test(Dataset):
    def __init__(self, dir, transform=None, norm=1):
        self.transform = transform
        self.n_pat = 0
        self.data_list = {}  # lista di tutti i pazienti
        self.norm = norm

        for root, _, fnames in sorted(os.walk(dir)):
            if norm==3:
             for fname in sorted(fnames):
                if fname.endswith('.npz'):
                    self.data_list[self.n_pat] = os.path.join(root, fname)
                    self.n_pat += 1
            else:
             for fname in fnames:
                if fname.endswith('.npz'):
                    self.data_list[self.n_pat] = os.path.join(root, fname)
                    self.n_pat += 1

    def __len__(self):
        length = self.n_pat
        return length

    def __getitem__(self, idx):

        print(self.data_list[idx])
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

class templatedataset_paired(Dataset):
    def __init__(self, dir, transform=None,data=0):
        self.transform = transform
        self.n_pat = 0
        self.stop = []
        self.lunghezze = []
        self.data_list = {}  # lista di tutti i pazienti
        self.seg_list = {}  # lista di tutti i pazienti
        self.data = data

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
        if self.data==0:
        	A = patient_data['data'][0, :, :, :]
        else:
        	A = patient_data['data'][1, :, :, :]

        A = np.rollaxis(A, 0, 3)
        shape = A.shape[:2]
        A = skimage.transform.resize(A, (128, 128))

        for i in range(A.shape[2]):
            A[:,:,i] = keep_largest(A[:,:,i])


        minimum = -74.40
        maximum = 308.58

        np.clip(A, minimum, maximum, out=A)
        A = (((A - minimum) / (maximum-minimum)) * 2.0) - 1.0


        A = A[np.newaxis, :, :,:]
        return torch.as_tensor(A, dtype=torch.float)
