import nibabel as nib
import numpy as np
import time
from skimage.transform import resize
import scipy.ndimage as ndimage
import skimage.measure as measure
import os
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
import sys
ee = sys.float_info.epsilon

def adjust_dim(image, dpatch_size, hpatch_size, wpatch_size):
    minimum = image.min()
    D, H, W = image.shape
    flag = 0
    if D<dpatch_size:
        x=dpatch_size-D
        a=np.zeros((x,H,W))
        a[:,:,:]=minimum
        if x>1:
            xx = int(round(x/2))
            new_image = np.concatenate((a[0:xx,:,:], image, a[xx:x, :, :]), axis=0)
        else:
            new_image=np.concatenate((image,a), axis=0)
        del image
        flag = 1
    if H<hpatch_size:
        y=hpatch_size-H
        if flag == 1:
            D, H, W = new_image.shape
            b = np.zeros((D, y, W))
            b[:,:,:]=minimum
            if y > 1:
                yy = int(round(y / 2))
                new_image2 = np.concatenate((b[:,0:yy,:], new_image, b[:,yy:y,:]), axis=1)
            else:
                new_image2 = np.concatenate((new_image, b), axis=1)
            del new_image
        else:
            b = np.zeros((D, y, W))
            b[:,:,:]=minimum
            if y > 1:
                yy = int(round(y / 2))
                new_image2 = np.concatenate((b[:, 0:yy, :], image, b[:, yy:y, :]), axis=1)
            else:
                new_image2 = np.concatenate((image, b), axis=1)
            del image
        flag = 2
    if W<wpatch_size:
        z=wpatch_size-W
        if flag == 1:
            D, H, W = new_image.shape
            c = np.zeros((D, H, z))
            c[:,:,:]=minimum
            if z > 1:
                zz = int(round(z / 2))
                new_image3 = np.concatenate((c[:,:,0:zz], new_image, c[:,:, zz:z]), axis=2)
            else:
                new_image3 = np.concatenate((new_image, c), axis=2)
            del new_image
        elif flag == 2:
            D, H, W = new_image2.shape
            c = np.zeros((D, H, z))
            c[:,:,:]=minimum
            if z > 1:
                zz = int(round(z / 2))
                new_image3 = np.concatenate((c[:, :, 0:zz], new_image2, c[:, :, zz:z]), axis=2)
            else:
                new_image3 = np.concatenate((new_image2, c), axis=2)
            del new_image2
        else:
            c=np.zeros((D,H,z))
            c[:,:,:]=minimum
            if z > 1:
                zz = int(round(z / 2))
                new_image3 = np.concatenate((c[:, :, 0:zz], image, c[:, :, zz:z]), axis=2)
            else:
                new_image3 = np.concatenate((image, c), axis=2)
            del image
        flag = 3


    #centrare l'immagine rispetto depth:
    if flag==1:
        final_image = new_image
    elif flag==2:
        final_image = new_image2
    elif flag == 3:
        final_image = new_image3
    else:
        final_image = image

    return final_image

def np_to_img(image):
    I8 = (((image - image.min()) / (image.max() - image.min() + ee)) * 255.0)
    img = Image.fromarray(I8.astype(np.uint8), 'L')
    return img

def keep_largest(image, mode = '2D'):
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
            if mode == '2D':
                labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
            else:
                labels_mask[rg.coords[:, 0], rg.coords[:, 1], rg.coords[:, 2]] = 0
    labels_mask[labels_mask != 0] = 1
    largest = labels_mask * image
    largest[labels_mask == 0] = image.min()
    return largest

def keep_largest_mask(image_original, minn = None, mode = '2D'):
    image = np.copy(image_original)
    if minn==None:
        minn = image.min()
    image[image > (minn + 0.005)] = 1
    image[image < 1] = 0
    xx = ndimage.morphology.binary_dilation(image, iterations=2).astype(int)  # default int=2
    xxx = ndimage.morphology.binary_fill_holes(xx).astype(int)
    labels_mask = measure.label(xxx)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            if mode == '2D':
                labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
            else:
                labels_mask[rg.coords[:, 0], rg.coords[:, 1], rg.coords[:, 2]] = 0
    labels_mask[labels_mask != 0] = 1
    mask = labels_mask
    return mask

def identify_black(patient):
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

def rescale(image):
    I8 = (((image - image.min()) / (image.max() - image.min())))
    return I8


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = im1 * im2

    return 2. * intersection.sum() / im_sum


start_time = time.time()
print('Upload reference image for pose and shape...')
#ref_path = '/home/glabarbera/cluster/nnunet/nnUNet_preprocessed/Task200_NECKER/nnUNetData_plans_v2.1_2D_stage0/Slices/NECKER_071_1601_311.npz'
#ref_img = np.load(ref_path)['arr_0'][0, :, :].astype(np.float32)
#ref_image = resize(ref_img, (512,512), order=1, mode='constant', cval=ref_img.min(), anti_aliasing=False)
ref_path = '/tsi/clusterhome/glabarbera/unet3d/nnUNet_preprocessed/Task200_NECKER/nnUNetData_plans_v2.1_stage1/NECKER_071_1601.npz'
ref_image = np.load(ref_path)['data'][0, :, :, :].astype(np.float32)
ref_image = adjust_dim(ref_image,512,512,512)
fixedImg = keep_largest(ref_image, mode = '3D')
fixedImage = keep_largest_mask(fixedImg, mode = '3D')
print(fixedImage.min(),fixedImage.max())
print(fixedImage.shape)
'''
a = np.load('/media/glabarbera/Donnees/Francesco/Database_GAN_128/trainA/H_O_Pancreas_2.npz', mmap_mode='r')
adata = a[a.files[0]]
print(adata.shape)
landmark1, landmark2 = identify_black(adata)  # calculate the number of black pixel
fixedImage = np.rollaxis(adata[:, :, landmark1:landmark2], 2,0)
fixedImage = rescale(np.rollaxis(fixedImage, 2,1))
print(fixedImage.min(),fixedImage.max())
print(fixedImage.shape)

b = np.load('/media/glabarbera/Donnees/Francesco/Database_GAN_128/trainB_new/1.6.1.4.1.9328.50.4.0667_N.npz', mmap_mode='r')
bdata = b[b.files[0]]
print(bdata.shape)
landmark1, landmark2 = identify_black(bdata)  # calculate the number of black pixel
movingImage = np.rollaxis(bdata[:, :, landmark1:landmark2], 2, 0)
movingImage = rescale(np.rollaxis(movingImage, 2,1))

print(movingImage.shape)
'''
preprocessing = [76.99, 67.73, 303.0, -36.0]
path = '/tsi/clusterhome/glabarbera/unet3d/nnUNet_raw_data_base/nnUNet_raw_data/Task200_NECKER/imagesTs'
#path = '/home/glabarbera/cluster/nnunet/nnUNet_preprocessed/Task200_NECKER/nnUNetData_plans_v2.1_2D_stage0/Slices/'
#path_new = '/home/glabarbera/cluster/nnunet/nnUNet_preprocessed/Task200_NECKER/nnUNetData_plans_v2.1_2D_stage0/Slices_registered_128/'
(_, _, filenames_imts) = next(os.walk(path))
filenames_imts = sorted(filenames_imts)
dice_array = np.zeros((len(filenames_imts)))
l2_array = np.zeros((len(filenames_imts)))
for i in range(len(filenames_imts)):
    print('Patient ',i)

    mov_img = nib.load(os.path.join(path, filenames_imts[i])).get_fdata()
    print(mov_img.shape)
    #for j in range(mov_img.shape[2]):
    #mov_image = mov_img[:,:,j]
    #mov_image = np.rollaxis(mov_image,1,0)
    if mov_img.shape[2]>512:
    	d= (mov_img.shape[2]-512)//2
    	mov_img = mov_img[:,:,d:d+512]
    mov_img = adjust_dim(mov_img,512,512,512)
    mov_image = np.rollaxis(mov_img,2,0)
    mov_image = np.rollaxis(mov_image,2,1)
    np.clip(mov_image, preprocessing[3], preprocessing[2], out=mov_image)
    mov_image = (mov_image - preprocessing[0]) / preprocessing[1]
    #mov_img = np.load(os.path.join(path, filenames_imts[i]))['arr_0'][0, :, :].astype(np.float32)
    #mov_image = resize(mov_img, (512, 512), order=1, mode='constant', cval=mov_img.min(), anti_aliasing=False)
    mov_Image = keep_largest(mov_image, mode = '3D')

    movingImage = keep_largest_mask(mov_Image, mode = '3D')
    print(movingImage.min(), movingImage.max())
    print(movingImage.shape)

    # Functional interface
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(fixedImage))
    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(movingImage))
    parameterMap = sitk.GetDefaultParameterMap("affine")
    parameterMap["Transform"] = ["SimilarityTransform"]
    #parameterMap["Metric"] = ["AdvancedKappaStatistic"]
    parameterMap['DefaultPixelValue'] = [str(movingImage.min())]
    elastixImageFilter.SetParameterMap(parameterMap)
    resultImage = sitk.GetArrayFromImage(elastixImageFilter.Execute())
    result_Image = np.copy(resultImage.astype(int))
    result_Image[result_Image>=1] = 1
    result_Image[result_Image<1] = 0

    del elastixImageFilter

    #np.savez_compressed(os.path.join(path_new, filenames_imts[i]), resultImage)
    print(result_Image.min(), result_Image.max())
    #maskres = keep_largest_mask(resultImage, movingImage.min())

    dice_array[i] = dice(fixedImage,result_Image)
    l2_array[i] = np.mean(np.abs(result_Image - fixedImage))
    '''
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(transformParameterMap)
    transformix.SetMovingImage(sitk.GetImageFromArray(mov_Image))
    transformix.Execute()
    result_Image = sitk.GetArrayFromImage(transformix.GetResultImage())
    '''
    if i==0:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('axial: affine')
        ax1.imshow(np_to_img(mov_Image[:,256,:]), cmap='gray', origin='lower')
        ax2.imshow(np_to_img(fixedImg[:,256,:]), cmap='gray', origin='lower')
        ax3.imshow(np_to_img(result_Image[:,256,:]), cmap='gray', origin='lower')
        fig.show()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('axial: affine')
        ax1.imshow(np_to_img(movingImage[:,256,:]), cmap='gray', origin='lower')
        ax2.imshow(np_to_img(fixedImage[:,256,:]), cmap='gray', origin='lower')
        ax3.imshow(np_to_img(result_Image[:,256,:]), cmap='gray', origin='lower')
        fig.show()

        #dice_array[i] /= mov_img.shape[2]
        #l2_array[i] /= mov_img.shape[2]
    print(dice_array)
    print(l2_array)
    end_time = time.time()
    print('TIME (s): ', end_time - start_time)
    print('TIME per subject (s): ', (end_time - start_time) / (i+1))

    del mov_image,movingImage,mov_Image,mov_img, result_Image

end_time = time.time()
print('TIME (s): ', end_time - start_time)
print('TIME per subject (s): ', (end_time - start_time)/15)

print('DICE AVERAGE: ',np.mean(dice_array), 'STD:', np.std(dice_array))
print('L2 AVERAGE: ',np.mean(l2_array), 'STD:', np.std(l2_array))


'''
path_old = '/home/glabarbera/cluster/nnunet/nnUNet_preprocessed/Task200_NECKER/nnUNetData_plans_v2.1_2D_stage0/Slices_registered/'
dice_array = np.zeros((len(filenames_imts)))
l1_array = np.zeros((len(filenames_imts)))
ref_path = '/home/glabarbera/cluster/nnunet/nnUNet_preprocessed/Task200_NECKER/nnUNetData_plans_v2.1_2D_stage0/Slices/NECKER_071_1601_311.npz'
ref_img = np.load(ref_path)['arr_0'][0, :, :].astype(np.float32)
ref_image = resize(ref_img, (512,512), order=1, mode='constant', cval=ref_img.min(), anti_aliasing=False)
fixedImage = keep_largest(ref_image)
mask = keep_largest_mask(fixedImage)
for i in range(len(filenames_imts)):
    res_img = np.load(os.path.join(path_old, filenames_imts[i]))['arr_0'].astype(np.float32)
    print(res_img.shape)
    resultImage = keep_largest(res_img)
    maskres = keep_largest_mask(resultImage, movingImage.min())
    dice_array[i] = dice(mask, maskres)
    print(dice_array[i])
    l1_array[i] = np.mean(np.abs(maskres - mask))
    print(l1_array[i])
print('DICE AVERAGE: ',np.mean(dice_array), 'STD:', np.std(dice_array))
print('L1 AVERAGE: ',np.mean(l1_array), 'STD:', np.std(l1_array))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('axial: affine')
ax1.imshow(np_to_img(movingImage[10,:,:]),cmap='gray')
ax2.imshow(np_to_img(fixedImage[10,:,:]),cmap='gray')
ax3.imshow(np_to_img(resultImage[10,:,:]),cmap='gray')
fig.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('coronal: affine')
ax1.imshow(np_to_img(movingImage[:,64,:]),cmap='gray')
ax1.set_ylim(0,500)
ax2.imshow(np_to_img(fixedImage[:,64,:]),cmap='gray')
ax2.set_ylim(0,500)
ax3.imshow(np_to_img(resultImage[:,64,:]),cmap='gray')
ax3.set_ylim(0,500)
fig.show()
fig.savefig('./plot0', bbox_inches='tight')


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('sagittal: affine')
ax1.imshow(np_to_img(movingImage[:,:,100]),cmap='gray')
ax1.set_ylim(0,500)
ax2.imshow(np_to_img(fixedImage[:,:,100]),cmap='gray')
ax2.set_ylim(0,500)
ax3.imshow(np_to_img(resultImage[:,:,100]),cmap='gray')
ax3.set_ylim(0,500)
fig.show()
fig.savefig('./plot1', bbox_inches='tight')



I8 = (((resultImage - resultImage.min()) / (resultImage.max() - resultImage.min())) * 255.0)
img = nib.Nifti1Image(I8,affine=np.eye(4))
nib.save(img,'/home/glabarbera/Desktop/elastix.nii.gz')
'''
