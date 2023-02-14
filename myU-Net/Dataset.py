import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from utils.cropping import crop_to_bbox, crop_to_nonzero
from utils.patches import adjust_dim_2d, adjust_dim
from utils.utils import keep_largest, keep_largest_mask
from skimage.transform import resize
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.abstract_transforms import Compose
from skimage.morphology import skeletonize
import sys
ee = sys.float_info.epsilon

class CT_Dataset(Dataset):
    """
    Our dataset class, inherited from the Dataset object from pytorch
    """

    def __init__(self, paths, patch_size, input_folder, channel_dim, in_c, tfms_train, tfms_train_seg, tfms_valid, massimo,
                 minimo, rsz, norm):
        self.paths = paths
        self.patch_size = patch_size
        self.channel_dim = channel_dim
        self.input_folder = input_folder
        self.tfms_train = tfms_train
        self.tfms_train_seg = tfms_train_seg
        self.tfms_valid = tfms_valid
        self.train_indices = []
        self.valid_indices = []
        self.rsz = rsz
        self.massimo = massimo
        self.minimo = minimo
        self.in_c = in_c
        
        if norm:
          #self.case = 4 #my norm with skel points
          self.case = 5 #my norm with a channel per structure with skel points
        else:
          #self.case = 0  # normal
          self.case = 1 #data_aug target
          #self.case = 2  #clip to max-min
          #self.case = 3  #rescale [-1,1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        images = np.load(self.paths[i])
        if len(self.patch_size) == 3:
            img = images[images.files[0]][0:self.in_c, :, :, :].astype(np.float32)
            seg = images[images.files[0]][-1, :, :, :].astype(np.long)
        else:
            img = images[images.files[0]][0:self.in_c, :, :].astype(np.float32)
            seg = images[images.files[0]][-1, :, :].astype(np.long)
            
        if self.input_folder == 'adults':
            img =  np.expand_dims(np.rollaxis(img[0],1,0),0)
            img = img[:,:,::-1]
            seg = np.rollaxis(seg,1,0)
            seg = seg[:,::-1]
        if len(self.patch_size) == 2:
            image = resize(img[0], (self.patch_size), order=1, mode='constant', cval=img.min(), anti_aliasing=False)
            image = np.expand_dims(image, 0)
            segmentation = resize(seg, (self.patch_size), order=0, mode='constant', cval=seg.min(), clip=True,  preserve_range=True, anti_aliasing=False)
            del img
            del seg
        else:
            if self.rsz:
                img = np.rollaxis(img[0],0,3)
                seg = np.rollaxis(seg,0,3)
                if img.shape[2]>img.shape[0]:
                    d = img.shape[2]-img.shape[0]
                    img2 = adjust_dim(img, img.shape[0]+d, img.shape[1]+d, img.shape[2])
                    seg2 = adjust_dim(seg, img.shape[0]+d, img.shape[1]+d, img.shape[2])

                elif img.shape[2]<img.shape[0]:
                    d = img.shape[0] - img.shape[2]
                    img2 = adjust_dim(img, img.shape[0], img.shape[1], img.shape[2] + d)
                    seg2 = adjust_dim(seg, img.shape[0], img.shape[1], img.shape[2] + d)

                else:
                    img2 = img
                    seg2 = seg

                image = resize(img2, (self.patch_size), order=1, mode='constant', cval=img2.min(), anti_aliasing=True)
                image = np.expand_dims(image, 0)
                segmentation = resize(seg2, (self.patch_size), order=0, mode='constant', cval=seg2.min(), clip=True, preserve_range=True,
                                      anti_aliasing=False)

                del img,seg,img2,seg2


            else:
                image = img
                segmentation = seg
                
        if self.case == 4:      
          dat = segmentation > 0
          skel = skeletonize(dat)
          if np.count_nonzero(skel)!=0:
            image = image * 175.44 + 188.93
            maskold = np.copy(image)
            maskold[skel==0] = np.nan
            image = (image - np.nanmean(maskold))/ np.nanstd(maskold) 
            del maskold
          del skel,dat
          image = np.expand_dims(np.expand_dims(image, 0), 0)  # we need to add batch and channels for transformations
          
        elif self.case == 5:
         image = image * 175.44 + 188.93
         images = np.zeros([1,self.channel_dim - 1, image.shape[0], image.shape[1], image.shape[2]]).astype(np.float32)
         for k in range(1, self.channel_dim):
             dat = segmentation == k
             skel = skeletonize(dat)
             if np.count_nonzero(skel)!=0:
                maskold = np.copy(image)
                maskold[skel==0] = np.nan
                images[0, k-1] = (image - np.nanmean(maskold)) / (np.nanstd(maskold) + 1)
                del maskold
             else:
             	images[0, k-1] = (image - 188.93) / 175.44
             del skel,dat
         image = images

        else:
            image = np.expand_dims(image, 0)  # we need to add batch for transformations
            segmentation = np.expand_dims(np.expand_dims(segmentation, 0), 0)
		
        
        if i in self.train_indices:
            if self.tfms_train is not None:
                data_dict = self.tfms_train(data=image, seg=segmentation)
                segmentation = data_dict['seg']
                segmentation[segmentation == -1] = 0
                somma = np.sum(segmentation)
                image = data_dict['data']
                if self.case == 1 and somma != 0:
                    image1 = image.copy()
                    data_dict_seg = self.tfms_train_seg(data=image, seg=segmentation)
                    image = data_dict_seg['data']
                    image1[np.where(segmentation != 0)] = image[np.where(segmentation != 0)]
                    image = image1

                	
        if i in self.valid_indices:
            if self.tfms_valid is not None:
                data_dict = self.tfms_valid(data=image, seg=segmentation)
                image = data_dict['data']
                segmentation = data_dict['seg']
                segmentation[segmentation == -1] = 0

        segmentation = np.squeeze(np.squeeze(segmentation, 0), 0)
        labels = np.zeros(np.append(self.channel_dim,  segmentation.shape)).astype(np.float16)
        
        for j in range(self.channel_dim):
            s = np.zeros(segmentation.shape).astype(np.float16)
            s[np.where(segmentation == j)] = 1
            labels[j] = s
            del s
        del segmentation
        
        if self.case == 2:
            np.clip(image, self.minimo, self.massimo, out=image)
            
        elif self.case == 3:
            image = ((image - self.minimo) / (
                        self.massimo - self.minimo) * 2.0) - 1.0  # [-1,+1] both median and mean about 0 and std dev 1
            # image = ((image-self.minimo)/(self.massimo-self.minimo)*1.0)  #[0,1] value same range as output segmentation

        if self.rsz:
            return torch.as_tensor(image, dtype=torch.float).squeeze(0), torch.as_tensor(labels, dtype=torch.long)
        else:
            return torch.as_tensor(image, dtype=torch.bfloat16).squeeze(0), torch.as_tensor(labels, dtype=torch.long)


class CT_Dataset_25D(Dataset):
    """
    Our dataset class, inherited from the Dataset object from pytorch
    """

    def __init__(self, paths, patch_size, input_folder, channel_dim, tfms_train, tfms_train_seg, tfms_valid, massimo,
                 minimo, rsz):
        self.paths = paths
        self.patch_size = patch_size
        self.channel_dim = channel_dim
        self.input_folder = input_folder
        self.tfms_train = tfms_train
        self.tfms_train_seg = tfms_train_seg
        self.tfms_valid = tfms_valid
        self.train_indices = []
        self.valid_indices = []
        self.rsz = rsz
        self.massimo = massimo
        self.minimo = minimo

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = np.load(self.paths[i])['arr_0'][0, :, :, :].astype(np.float32)
        segmentation = np.load(self.paths[i])['arr_0'][1, :, :, :].astype(np.long)

        image = np.expand_dims(image, 0)  # we need to add batch
        segmentation = np.expand_dims(segmentation, 0)
        if i in self.train_indices:
            if self.tfms_train is not None:
                data_dict = self.tfms_train(data=image, seg=segmentation)
                segmentation = data_dict['seg']
                segmentation[segmentation == -1] = 0

                image = data_dict['data']

            if self.tfms_valid is not None:
                data_dict = self.tfms_valid(data=image, seg=segmentation)
                image = data_dict['data']
                segmentation = data_dict['seg']
                segmentation[segmentation == -1] = 0
        segmentation = np.squeeze(segmentation, 0)
        return torch.as_tensor(image, dtype=torch.bfloat16).squeeze(0), torch.as_tensor(segmentation, dtype=torch.long)


def Prepare_Dataset(paths, patch_size, input_folder, channel_dim, in_c, batch_size, workers, dda, massimo, minimo, network,
                    rsz=False, norm=False):
    #if input_folder == 'adults':
    #    r_crop = True
    #else:
    r_crop = False
    # parameters for the data augmentation!
    spatial_transform = SpatialTransform_2(
        patch_size=patch_size,
        patch_center_dist_from_border=patch_size // 2,
        do_elastic_deform=False,
        deformation_scale=(0, 0.25),
        do_rotation=dda,
        angle_x=((-30. / 360 * 2. * np.pi), (30. / 360 * 2. * np.pi)),
        angle_y=((-30. / 360 * 2. * np.pi), (30. / 360 * 2. * np.pi)),
        angle_z=((-30. / 360 * 2. * np.pi), (30. / 360 * 2. * np.pi)),
        do_scale=dda,
        scale=(0.7, 1.4),
        border_mode_data='constant',
        border_cval_data=minimo,
        order_data=3,
        border_mode_seg='constant',
        border_cval_seg=0,
        order_seg=0,
        random_crop=r_crop,
        data_key='data',
        label_key='seg',
        p_el_per_sample=0.2,
        p_scale_per_sample=0.2,
        p_rot_per_sample=0.2)

    spatial_transform_valid = SpatialTransform_2(
        patch_size=patch_size,
        patch_center_dist_from_border=patch_size // 2,
        do_elastic_deform=False,
        do_rotation=False,
        do_scale=False,
        border_mode_data='constant',
        border_cval_data=minimo,
        order_data=3,
        border_mode_seg='constant',
        border_cval_seg=0,
        order_seg=0,
        random_crop=r_crop,
        data_key='data',
        label_key='seg',
        p_el_per_sample=0,
        p_scale_per_sample=0,
        p_rot_per_sample=0)
    # Technically we are still seing different images each time, because we have ramdom crop,
    # but I think that this is good enough as an estimation of our loss...

    gaussian_noise = GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_sample=0.15, per_channel=True)
    gaussian_blur = GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True, p_per_channel=0.5,
                                          p_per_sample=0.2)
    brightness = BrightnessMultiplicativeTransform(multiplier_range=(0.7, 1.3), p_per_sample=0.15, per_channel=True)
    contrast = ContrastAugmentationTransform(contrast_range=(0.65, 1.5), p_per_sample=0.15, per_channel=True)
    low_res = SimulateLowResolutionTransform(zoom_range=(1, 2), per_channel=True,
                                             p_per_channel=0.5,
                                             order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                             ignore_axes=None)
    gamma = GammaTransform(gamma_range=(0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True,
                           p_per_sample=0.15)
    if len(patch_size) == 3:
        mirror = MirrorTransform(axes=(0, 1, 2))
    else:
        mirror = MirrorTransform(axes=(0, 1))
    # -------------------------------------------------------------------------------------------------------------
    if dda:
        tfms_train = [spatial_transform, gaussian_noise, gaussian_blur, brightness, contrast, low_res, gamma, mirror]
        tfms_train_seg = [gaussian_noise, gaussian_blur, brightness, contrast, low_res, gamma]
        tfms_valid = [spatial_transform_valid]
    else:
        tfms_train = [spatial_transform, gaussian_noise, gaussian_blur, brightness, contrast, low_res, gamma]
        tfms_train_seg = []#gaussian_noise, gaussian_blur, brightness, contrast, low_res, gamma]
        tfms_valid = [spatial_transform_valid]

    tfms_train = Compose(tfms_train)
    tfms_train_seg = Compose(tfms_train_seg)
    tfms_valid = Compose(tfms_valid)

    if network == 'nnunet2.5D':
        dataset = CT_Dataset_25D(paths, patch_size, input_folder, channel_dim, tfms_train, tfms_train_seg, tfms_valid,
                                 massimo, minimo, rsz, norm)
    else:
        dataset = CT_Dataset(paths, patch_size, input_folder, channel_dim, in_c, tfms_train, tfms_train_seg, tfms_valid,
                             massimo, minimo, rsz, norm)

    torch.manual_seed(7)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [int(0.9 * len(dataset)),
                                                                (len(dataset) - int(0.9 * len(dataset)))])
    dataset.train_indices = train_dataset.indices
    dataset.valid_indices = val_dataset.indices

    print('train dataset: ', len(train_dataset), 'val dataset: ', len(val_dataset))

    # example of img from custom dataset
    # n_img = 0
    # image, mask = train_dataset[n_img]
    # print('image size:', image.size(), image.dtype, torch.max(image), torch.min(image))
    # print('mask size:', mask.size(), mask.dtype, torch.max(mask), torch.min(mask))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, val_loader


class CustomDataset_new(Dataset):
    def __init__(self, patch_ids, imaget, in_c, dpatch_size, hpatch_size, wpatch_size, do_seg=False, segt=None):  # initial logic happens like transform
        self.patch_ids = patch_ids
        self.imaget = imaget
        self.dpatch_size = dpatch_size
        self.hpatch_size = hpatch_size
        self.wpatch_size = wpatch_size
        self.do_seg = do_seg
        self.segt = segt
        self.in_c = in_c

    def __getitem__(self, index):
        (d, h, w) = self.patch_ids[index]
        image = self.imaget[0:self.in_c, d:d + self.dpatch_size, h:h + self.hpatch_size, w:w + self.wpatch_size]
        t_image = torch.from_numpy(image).type('torch.FloatTensor')
        if self.do_seg:
            mask = self.segt[:, d:d + self.dpatch_size, h:h + self.hpatch_size, w:w + self.wpatch_size]
            t_mask = torch.from_numpy(mask).type('torch.FloatTensor')
            return t_image, t_mask
        else:
            return t_image

    def __len__(self):  # return count of sample we have
        return len(self.patch_ids)


def Prepare_Test_Data_new(patch_ids, imaget, in_c, dpatch_size, hpatch_size, wpatch_size, batch_size, workers, do_seg=False,segt=None):
    test_dataset = CustomDataset_new(patch_ids, imaget, in_c, dpatch_size, hpatch_size, wpatch_size, do_seg, segt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=workers)

    return test_loader
    
class CustomDataset_skel(Dataset):
    def __init__(self, patch_ids, imaget, dpatch_size, hpatch_size, wpatch_size, segt=None,
                 rsz=False, norm = False):  # initial logic happens like transform
        self.patch_ids = patch_ids
        self.imaget = imaget
        self.dpatch_size = dpatch_size
        self.hpatch_size = hpatch_size
        self.wpatch_size = wpatch_size
        self.segt = segt
        self.rsz = rsz
        if norm:
          #self.case = 1
          self.case = 2
        else:
          self.case = 0

    def __getitem__(self, index):
        (d, h, w) = self.patch_ids[index]
        image = np.full((32, 64, 64), self.imaget.min(), dtype=np.float32)
        patch =  self.imaget[d-16:d+16,h-32:h+32,w-32:w+32]
        x,y,z = 16-patch.shape[0]//2, 32-patch.shape[1]//2, 32-patch.shape[2]//2
        image[x:x+patch.shape[0],y:y+patch.shape[1],z:z+patch.shape[2]] = patch
        
        mask = np.zeros((self.segt.shape[0],32, 64, 64), dtype=np.float32)
        segpatch = self.segt[:, d - 16:d + 16, h - 32:h + 32, w - 32:w + 32]
        mask[:,x:x+patch.shape[0],y:y+patch.shape[1],z:z+patch.shape[2]] = segpatch
        
        if self.case == 1:
              y = np.argmax(mask, axis=0)
              dat = y > 0
              skel = skeletonize(dat)
              if np.count_nonzero(skel)!=0:
                image = image * 175.44 + 188.93
                maskold = np.copy(image)
                maskold[skel==0] = np.nan
                image = (image - np.nanmean(maskold))/ np.nanstd(maskold)
                del maskold
              del dat,skel,y
              t_image = torch.from_numpy(image).type('torch.FloatTensor').unsqueeze(0)
          
        elif self.case == 2:
             y = np.argmax(mask, axis=0)
             image = image * 175.44 + 188.93
             images = np.zeros([mask.shape[0] - 1, image.shape[0], image.shape[1], image.shape[2]]).astype(np.float32)
             for k in range(1, mask.shape[0]):
                 dat = y == k
                 skel = skeletonize(dat)
                 if np.count_nonzero(skel)!=0:
                    maskold = np.copy(image)
                    maskold[skel==0] = np.nan
                    images[k-1] = (image - np.nanmean(maskold)) / (np.nanstd(maskold) + 1)
                    del maskold
                 else:
                    images[k-1] = (image - 188.93) / 175.44
                 del skel,dat
             image = images
             t_image = torch.from_numpy(image).type('torch.FloatTensor')

        else:
            t_image = torch.from_numpy(image).type('torch.FloatTensor').unsqueeze(0)
         

        t_mask = torch.from_numpy(mask).type('torch.FloatTensor')

        return t_image, t_mask

    def __len__(self):  # return count of sample we have
        return len(self.patch_ids)
        
def Prepare_Test_Data_skel(patch_ids, imaget, dpatch_size, hpatch_size, wpatch_size, batch_size, workers, 
                          rsz=False, segt=None, norm=False):
    test_dataset = CustomDataset_skel(patch_ids, imaget, dpatch_size, hpatch_size, wpatch_size, segt, rsz, norm)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=workers)

    return test_loader




def Prepare_Dataset_train(paths, patch_size, input_folder, channel_dim, in_c, batch_size, workers, dda, massimo, minimo, network,
                    rsz=False, norm=False):
    if input_folder == 'adults':
        r_crop = True
    else:
        r_crop = False
    # parameters for the data augmentation!
    spatial_transform = SpatialTransform_2(
        patch_size=patch_size,
        patch_center_dist_from_border=patch_size // 2,
        do_elastic_deform=False,
        deformation_scale=(0, 0.25),
        do_rotation=dda,
        angle_x=((-30. / 360 * 2. * np.pi), (30. / 360 * 2. * np.pi)),
        angle_y=((-30. / 360 * 2. * np.pi), (30. / 360 * 2. * np.pi)),
        angle_z=((-30. / 360 * 2. * np.pi), (30. / 360 * 2. * np.pi)),
        do_scale=dda,
        scale=(0.7, 1.4),
        border_mode_data='constant',
        border_cval_data=minimo,
        order_data=3,
        border_mode_seg='constant',
        border_cval_seg=0,
        order_seg=0,
        random_crop=r_crop,
        data_key='data',
        label_key='seg',
        p_el_per_sample=0.2,
        p_scale_per_sample=0.2,
        p_rot_per_sample=0.2)

    spatial_transform_valid = SpatialTransform_2(
        patch_size=patch_size,
        patch_center_dist_from_border=patch_size // 2,
        do_elastic_deform=False,
        do_rotation=False,
        do_scale=False,
        border_mode_data='constant',
        border_cval_data=minimo,
        order_data=3,
        border_mode_seg='constant',
        border_cval_seg=0,
        order_seg=0,
        random_crop=r_crop,
        data_key='data',
        label_key='seg',
        p_el_per_sample=0,
        p_scale_per_sample=0,
        p_rot_per_sample=0)
    # Technically we are still seing different images each time, because we have ramdom crop,
    # but I think that this is good enough as an estimation of our loss...

    gaussian_noise = GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_sample=0.15, per_channel=True)
    gaussian_blur = GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                          p_per_channel=0.5,
                                          p_per_sample=0.2)
    brightness = BrightnessMultiplicativeTransform(multiplier_range=(0.7, 1.3), p_per_sample=0.15, per_channel=True)
    contrast = ContrastAugmentationTransform(contrast_range=(0.65, 1.5), p_per_sample=0.15, per_channel=True)
    low_res = SimulateLowResolutionTransform(zoom_range=(1, 2), per_channel=True,
                                             p_per_channel=0.5,
                                             order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                             ignore_axes=None)
    gamma = GammaTransform(gamma_range=(0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True,
                           p_per_sample=0.15)
    if len(patch_size) == 3:
        mirror = MirrorTransform(axes=(0, 1, 2))
    else:
        mirror = MirrorTransform(axes=(0, 1))
    # -------------------------------------------------------------------------------------------------------------
    if dda:
        tfms_train = [spatial_transform, gaussian_noise, gaussian_blur, brightness, contrast, low_res, gamma, mirror]
        tfms_train_seg = [gaussian_noise, gaussian_blur, brightness, contrast, low_res, gamma]
        tfms_valid = [spatial_transform_valid]
    else:
        tfms_train = [spatial_transform, gaussian_noise, gaussian_blur, brightness, contrast, low_res, gamma]
        tfms_train_seg = [gaussian_noise, gaussian_blur, brightness, contrast, low_res, gamma]
        tfms_valid = [spatial_transform_valid]

    tfms_train = Compose(tfms_train)
    tfms_train_seg = Compose(tfms_train_seg)
    tfms_valid = Compose(tfms_valid)

    dataset = CT_Dataset(paths, patch_size, input_folder, channel_dim, in_c, tfms_train, tfms_train_seg, tfms_valid,
                             massimo, minimo, rsz, norm)

    train_dataset, _ = torch.utils.data.random_split(dataset,[int(len(dataset)), 0])
    dataset.train_indices = train_dataset.indices

    print('train dataset: ', len(train_dataset))


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    return train_loader

def Prepare_Dataset_val(paths, patch_size, input_folder, channel_dim, in_c, batch_size, workers, dda, massimo, minimo, network,
                    rsz=False, norm=False):
    if input_folder == 'adults':
        r_crop = True
    else:
        r_crop = False
    # parameters for the data augmentation!


    spatial_transform = SpatialTransform_2(
        patch_size=patch_size,
        patch_center_dist_from_border=patch_size // 2,
        do_elastic_deform=False,
        deformation_scale=(0, 0.25),
        do_rotation=dda,
        angle_x=((-30. / 360 * 2. * np.pi), (30. / 360 * 2. * np.pi)),
        angle_y=((-30. / 360 * 2. * np.pi), (30. / 360 * 2. * np.pi)),
        angle_z=((-30. / 360 * 2. * np.pi), (30. / 360 * 2. * np.pi)),
        do_scale=dda,
        scale=(0.7, 1.4),
        border_mode_data='constant',
        border_cval_data=minimo,
        order_data=3,
        border_mode_seg='constant',
        border_cval_seg=0,
        order_seg=0,
        random_crop=r_crop,
        data_key='data',
        label_key='seg',
        p_el_per_sample=0.2,
        p_scale_per_sample=0.2,
        p_rot_per_sample=0.2)

    spatial_transform_valid = SpatialTransform_2(
        patch_size=patch_size,
        patch_center_dist_from_border=patch_size // 2,
        do_elastic_deform=False,
        do_rotation=False,
        do_scale=False,
        border_mode_data='constant',
        border_cval_data=minimo,
        order_data=3,
        border_mode_seg='constant',
        border_cval_seg=0,
        order_seg=0,
        random_crop=r_crop,
        data_key='data',
        label_key='seg',
        p_el_per_sample=0,
        p_scale_per_sample=0,
        p_rot_per_sample=0)


    gaussian_noise = GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_sample=0.15, per_channel=True)
    gaussian_blur = GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True, p_per_channel=0.5,
                                          p_per_sample=0.2)
    brightness = BrightnessMultiplicativeTransform(multiplier_range=(0.7, 1.3), p_per_sample=0.15, per_channel=True)
    contrast = ContrastAugmentationTransform(contrast_range=(0.65, 1.5), p_per_sample=0.15, per_channel=True)
    low_res = SimulateLowResolutionTransform(zoom_range=(1, 2), per_channel=True,
                                             p_per_channel=0.5,
                                             order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                             ignore_axes=None)
    gamma = GammaTransform(gamma_range=(0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True,
                           p_per_sample=0.15)
    if len(patch_size) == 3:
        mirror = MirrorTransform(axes=(0, 1, 2))
    else:
        mirror = MirrorTransform(axes=(0, 1))
    # -------------------------------------------------------------------------------------------------------------
    if dda:
        tfms_train = [spatial_transform, gaussian_noise, gaussian_blur, brightness, contrast, low_res, gamma, mirror]
        tfms_train_seg = [gaussian_noise, gaussian_blur, brightness, contrast, low_res, gamma]
        tfms_valid = [spatial_transform_valid]
    else:
        tfms_train = [spatial_transform, gaussian_noise, gaussian_blur, brightness, contrast, low_res, gamma]
        tfms_train_seg = [gaussian_noise, gaussian_blur, brightness, contrast, low_res, gamma]
        tfms_valid = [spatial_transform_valid]

    tfms_train = Compose(tfms_train)
    tfms_train_seg = Compose(tfms_train_seg)
    tfms_valid = Compose(tfms_valid)

    dataset = CT_Dataset(paths, patch_size, input_folder, channel_dim, in_c, tfms_train, tfms_train_seg, tfms_valid,
                             massimo, minimo, rsz, norm)
    _, val_dataset = torch.utils.data.random_split(dataset,[int(0.9 * len(dataset)),(len(dataset) - int(0.9 * len(dataset)))])
    dataset.valid_indices = val_dataset.indices

    print('val dataset: ', len(val_dataset))

    # example of img from custom dataset
    # n_img = 0
    # image, mask = train_dataset[n_img]
    # print('image size:', image.size(), image.dtype, torch.max(image), torch.min(image))
    # print('mask size:', mask.size(), mask.dtype, torch.max(mask), torch.min(mask))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return val_loader


class CustomDataset_crop(Dataset):
    def __init__(self, imaget, in_c, dpatch_size, hpatch_size, wpatch_size, segt=None):  # initial logic happens like transform
        self.imaget = imaget
        self.patch_size = (dpatch_size, hpatch_size, wpatch_size)
        self.segt = segt
        self.in_c = in_c

    def __getitem__(self, index):

        img = self.imaget[0, :, :, :]
        #img = np.rollaxis(img, 0, 3)
        if img.shape[0] > img.shape[2]:
            d = (img.shape[0] - 512) // 2
            img2 = img[d:d + 512,:,:]

        elif img.shape[0] < img.shape[2]:
            d = img.shape[2] - img.shape[0]
            img2 = adjust_dim(img, img.shape[0]+d, img.shape[1], img.shape[2])

        else:
            img2 = img
        print(img2.shape)
        image = keep_largest(resize(img2, (self.patch_size), order=1, mode='constant', cval=img2.min(), anti_aliasing=True), mode = '3D')
        pose = keep_largest_mask(image, mode = '3D')
        image = np.expand_dims(image, 0)
        pose = np.expand_dims(pose,0)
        img2 = np.expand_dims(img2, 0)
        #seg2 = np.expand_dims(seg2, 0)
        del img

        if self.segt.any() != None:
            seg = np.argmax(self.segt, axis=0)
            #seg = np.rollaxis(seg, 0, 3)
            if seg.shape[0] > seg.shape[2]:
                seg2 = seg[d:d + 512,:, :]

            elif seg.shape[0] < seg.shape[2]:
                seg2 = adjust_dim(seg, img.shape[0]+d, img.shape[1], img.shape[2])

            else:
                seg2 = seg

            del seg

            t_mask = torch.from_numpy(seg2).type('torch.FloatTensor')

        else:
            t_mask = torch.zeros(1)

        t_img2 = torch.from_numpy(img2).type('torch.FloatTensor')
        t_pose = torch.from_numpy(pose).type('torch.FloatTensor')
        t_image = torch.from_numpy(image).type('torch.FloatTensor')

        return t_image, t_mask, t_img2, t_pose


    def __len__(self):  # return count of sample we have
        return 1


def Prepare_Test_Data_crop(imaget, in_c, dpatch_size, hpatch_size, wpatch_size, batch_size, workers, segt=None):
    test_dataset = CustomDataset_crop(imaget, in_c, dpatch_size, hpatch_size, wpatch_size, segt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=workers)

    return test_loader

