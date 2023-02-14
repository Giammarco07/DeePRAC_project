from PIL import Image
import numpy as np
import sys
ee = sys.float_info.epsilon

def np_to_img(I, mode, massimo = None, minimo = None):
    if mode == 'image':
        if massimo == None:
            massimo = I.max()
        if minimo == None:
            minimo = I.min()
        I8 = (((I - minimo) / (massimo - minimo)) * 255.0).astype(np.uint8)
    elif mode == 'target':
        I8 = (((I - I.min()) / (3.0 - I.min() + ee)) * 255.0).astype(np.uint8)
    img = Image.fromarray(I8)

    return img

def ensemble(out, channel_dim=2):
    s1 = out[:,1].clone().unsqueeze(1)
    s1[s1<=0.5] = 0
    s1[s1 > 0.5] = 1
    out_ = s1
    if channel_dim >2:
        s2 = out[:,2].clone().unsqueeze(1)
        s2[s2 <= 0.5] = 0
        s2[s2 > 0.5] = 2
        out_ += s2
        out_[out_>2]=1
    if channel_dim>3:
        s3 = out[:,3].clone().unsqueeze(1)
        s3[s3 <= 0.5] = 0
        s3[s3 > 0.5] = 3
        out_ += s3
        out_[out_>3]=1
    if channel_dim>4:
        print('WARNING! For now just segmentation of 3 structures is supported!')

    return out_

def ensemble_np(out, channel_dim=2):
    s1 = np.copy(out[1])
    s1[s1<=0.5] = 0
    s1[s1 > 0.5] = 1
    out_ = s1
    if channel_dim >2:
        s2 = np.copy(out[2])
        s2[s2 <= 0.5] = 0
        s2[s2 > 0.5] = 2
        out_ += s2
        #out_[out_>2]=2.5
    if channel_dim>3:
        s3 = np.copy(out[3])
        s3[s3 <= 0.5] = 0
        s3[s3 > 0.5] = 3
        out_ += s3
        out_[out_>3]=1
    if channel_dim>4:
        print('WARNING! For now just segmentation of 3 structures is supported!')

    return out_