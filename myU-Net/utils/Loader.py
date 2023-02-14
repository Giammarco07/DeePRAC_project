import numpy as np
import random
from utils.cropping import crop_to_nonzero

def labels(y):
    flag = 0
    kidney = np.sum(y[0,:,:,:])
    tumor = np.sum(y[1,:,:,:])
    if kidney != 0 and tumor != 0:
        flag = 1
    return flag

def labels_2d(y):
    flag = 0
    kidney = np.sum(y[0,:,:])
    tumor = np.sum(y[1,:,:])
    if kidney != 0 or tumor != 0:
        flag = 1
    return flag


def data_loader(path, max):
    i = 0
    x_patches = np.zeros((500, 80, 160, 160, 1))
    y_patches = np.zeros((500, 80, 160, 160, 2))
    samples = []
    while i<500:
        r = random.randint(0, max-1)
        if r not in samples:
            n = np.load(path + '/'+ str(r) + '.npz')
            x = n['x']
            y = n['y']
            if i<166:
                flag = labels(y)
                if flag == 1:
                    x_patches[i] = x
                    y_patches[i] = y
                    i += 1
                    samples.append(r)
            else:
                x_patches[i] = x
                y_patches[i] = y
                i += 1
                samples.append(r)

    indices = np.arange(x_patches.shape[0])
    np.random.shuffle(indices)

    x_patches = x_patches[indices]
    y_patches = y_patches[indices]

    return np.asarray(x_patches, dtype=np.float32), np.asarray(y_patches, dtype=np.float32)

def data_loader_path(path, num, max):
    i = 0
    samples = []
    all_images_path = []
    meta = 0
    while i<num:
        r = random.randint(100, max-1)
        if r not in samples:
            image_path = path + '/'+ str(r) + '.npz'
            n = np.load(image_path)
            y = n['y']
            if meta == 0:
                flag = labels(y)
                if flag == 1:
                    all_images_path.append(image_path)
                    i += 1
                    meta = 1
                    samples.append(r)
            else:
                all_images_path.append(image_path)
                i += 1
                meta = 0
                samples.append(r)

    return all_images_path


def foreground(y, channel_dim):
    flag = 0
    s1 = np.sum(y[np.where(y==1)])
    if s1!=0:
        flag=1
    if channel_dim>2:
        s2 = np.sum(y[np.where(y==2)])
        if s2!=0:
            flag = 2
    if channel_dim>3:
        s3 = np.sum(y[np.where(y==3)])
        if s3!=0:
            flag = 3
    return flag

def foreground_2(y, channel_dim, min_pix):
    flag = np.zeros((channel_dim-1))
    for i in range(channel_dim-1):
        s = np.sum(y[np.where(y==(i+1))])
        if s>min_pix:
            flag[i] = 1

    return flag



def get_image_file(path, channel_dim, patch_size, no_background = False):
    """
    From a Path object (pathlib), the function return all the files in this path

    input:
        - path: Path object (from pathlib)
    returns:
        - files: all the files inside path, sorted alphabetically
    """
    paths = []
    paths_background = []
    b = 0
    paths_s1 = []
    s1 = 0
    if channel_dim>2:
        paths_s2 = []
        s2 = 0
    if channel_dim>3:
        paths_s3 = []
        s3 = 0
    for i in path.iterdir():
        if (i.is_file()) & (i.suffix == '.npz'):
                if len(patch_size)==3:
                    seg = np.load(i)['arr_0'][1, :, :, :].astype(np.float32)
                else:
                    seg = np.load(i)['arr_0'][1, :, :].astype(np.float32)
                flag = foreground(seg,channel_dim)
                if flag>0:
                    if flag==1:
                        paths_s1.append(i)
                    elif flag==2:
                        paths_s2.append(i)
                    else:  #flag==3
                        paths_s3.append(i)
                else: #flag==0
                    paths_background.append(i)

    if no_background:
        print('WARNING! No images with all backgrounds will be used!')
        num = len(paths_s1)
        print('s1:', len(paths_s1))
    else:
        num = max(len(paths_s1),len(paths_background))
        print('background:', len(paths_background))
        print('s1:', len(paths_s1))
    if channel_dim>2:
        num = max(num, len(paths_s2))
        print('s2:', len(paths_s2))
    if channel_dim>3:
        num = max(num, len(paths_s3))
        print('s3:', len(paths_s3))
    for i in range(num):
        if no_background==False:
            if i < len(paths_background):
                paths.append(paths_background[i])
            elif b < len(paths_background):
                paths.append(paths_background[b])
                b += 1
            else:
                b = 0
                paths.append(paths_background[b])
                b += 1

        if i < len(paths_s1):
            paths.append(paths_s1[i])
        elif s1 < len(paths_s1):
            paths.append(paths_s1[s1])
            s1 += 1
        else:
            s1 = 0
            paths.append(paths_s1[s1])
            s1 += 1

        if channel_dim>2:
            if i < len(paths_s2):
                paths.append(paths_s2[i])
            elif s2 < len(paths_s2):
                paths.append(paths_s2[s2])
                s2 +=1
            else:
                s2 =0
                paths.append(paths_s2[s2])
                s2 += 1

        if channel_dim>3:
            if i < len(paths_s3):
                paths.append(paths_s3[i])
            elif s3 < len(paths_s3):
                paths.append(paths_s3[s3])
                s3 += 1
            else:
                s3 = 0
                paths.append(paths_s3[s3])
                s3 += 1

    return paths


def get_image_file_2(path, channel_dim, patch_size, min_pix, tot=False):
    """
    From a Path object (pathlib), the function return all the files in this path

    input:
        - path: Path object (from pathlib)
    returns:
        - files: all the files inside path, sorted alphabetically
    """
    n = 0
    flags = np.array([])
    mat = np.zeros((channel_dim - 1))
    count = np.zeros((2 ** (channel_dim - 1)))
    paths_b = []
    paths = []
    for i in path.iterdir():
        if (i.is_file()) & (i.suffix == '.npz'):
            images = np.load(i)
            if len(patch_size) == 3:
                seg = images[images.files[0]][-1, :, :, :].astype(np.float32)
            else:
                seg = images[images.files[0]][-1, :, :].astype(np.float32)
            flag = foreground_2(seg, channel_dim, min_pix)
            mat = mat + flag
            num = int(np.sum((2 ** np.arange(flag.size)[::-1]) * flag))
            count[num] += 1  # --> array to binary
            paths_b.append(i)
            if num != 0:
                n += 1
                flags = np.append(flags, num)
                paths.append(i)

    print(count)
    if channel_dim > 2:
        print(
            'More than 1 structures --> OVERSAMPLING will be applied to have equal prob for each of the foreground classes')
        matmax = mat.max()
        print('max: ', matmax, 'in s', mat.argmax())
        for i in range(channel_dim - 1):
            print('s' + str(i + 1) + ' appears in #images:', mat[i])
            j = 0
            p = 0
            while mat[i] != matmax:
                if flags[j] == (2 ** (channel_dim - 2 - i)):
                    paths.append(paths[j])
                    mat[i] += 1
                    p = 1
                j += 1
                if j == n:
                    j = 0
                    if p == 0:
                    	print('no patches with just this label exist!')
                    	break
            print('now s' + str(i + 1) + ' appears in #images:', mat[i])
    else:
        print('s1 appears in #images:', count[1])

    num = min(len(paths_b), len(paths))
    if tot:
    	num = max(len(paths_b), len(paths))

    return paths_b, paths, num

def get_image_file_3(path, channel_dim, patch_size):
    """
    From a Path object (pathlib), the function return all the files in this path

    input:
        - path: Path object (from pathlib)
    returns:
        - files: all the files inside path, sorted alphabetically
    """
    paths = []
    paths_s1 = []
    s1 = 0
    if channel_dim>2:
        paths_s2 = []
    if channel_dim>3:
        paths_s3 = []
    for i in path.iterdir():
        if (i.is_file()) & (i.suffix == '.npz'):
                if len(patch_size)==3:
                    seg = np.load(i)['arr_0'][1, :, :, :].astype(np.float32)
                else:
                    seg = np.load(i)['arr_0'][1, :, :].astype(np.float32)
                s1 = np.sum(seg[np.where(seg == 1)])
                if s1 != 0:
                    paths_s1.append(i)
                if channel_dim > 2:
                    s2 = np.sum(seg[np.where(seg == 2)])
                    if s2 != 0:
                        paths_s2.append(i)
                if channel_dim > 3:
                    s3 = np.sum(seg[np.where(seg == 3)])
                    if s3 != 0:
                        paths_s3.append(i)
                paths.append(i)

    num = len(paths_s1)
    print('s1:', len(paths_s1))
    if channel_dim > 2:
        num = min(num, len(paths_s2))
        print('s2:', len(paths_s2))
    if channel_dim > 3:
        num = min(num, len(paths_s3))
        print('s3:', len(paths_s3))

    if channel_dim == 3:
        return paths, paths_s1, paths_s2, num
    elif channel_dim == 4:
        return paths, paths_s1, paths_s2, paths_s3, num
    else:
        return paths, paths_s1, num

def foreground_4(y):
    flag = 0
    kidney = np.sum(y[np.where(y==1)])
    tumor = np.sum(y[np.where(y==2)])
    if kidney!=0 or tumor!=0:
        flag = 1
    return flag


def get_image_file_4(path):
    """
    From a Path object (pathlib), the function return all the files in this path

    input:
        - path: Path object (from pathlib)
    returns:
        - files: all the files inside path, sorted alphabetically
    """
    paths_foreground = []
    paths_background = []
    paths = []
    for i in path.iterdir():
        if (i.is_file()) & (i.suffix == '.npz'):
                image = np.load(i)['arr_0'][0, :, :].astype(np.float32)
                seg = np.load(i)['arr_0'][1, :, :].astype(np.float32)
                img,_ = crop_to_nonzero(image,image.min())
                flag = foreground_4(seg)
                if flag == 1 and img.shape[0] <= 512:
                    paths_foreground.append(i)
                elif flag==0 and img.shape[0] <= 512:
                    paths_background.append(i)
    len_fore = len(paths_foreground)
    print(len(paths_foreground))
    len_back = len(paths_background)
    print(len(paths_background))
    for i in range(min(len_fore,len_back)):
        paths.append(paths_foreground[i])
        paths.append(paths_background[i])

    return paths
