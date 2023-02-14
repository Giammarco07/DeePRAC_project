from functools import singledispatch

import numpy as np

from fastai_function import slice_sampler,slice_sampler_normalized, NumpyNumpyList, NumpyImageList
from fastai.vision import *
from fastai.basics import *
class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = torch.nn.Sequential(*(list(module.children())[:-4]))
        self.lin = nn.Conv2d(128, 1, 1, stride=1)
        #self.linear = nn.Sequential(nn.Linear(16*16,1),nn.Tanh())
        #self.output = nn.Tanh()

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, 1,  x.size(2), x.size(3)).repeat(1, 3, 1, 1)
        x = self.module(x)
        x = self.lin(x)
        #x = self.output(self.lin(x))
        #x = x.view(t * n,1,-1)
        #x = self.linear(x)
        x = x.mean([2, 3])
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr



# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
m = SequenceWise(models.resnet34(pretrained=False)).to(device)
print(m)
state = torch.load('./models/classifier-model-saved-by-callback.pth')
print(state)
m.cuda().load_state_dict(state['model'])
torch.save(m, './ssbr_abdominal_ct')

net = torch.load('./ssbr_abdominal_ct').cuda()
net.eval()
#net = m
# %%



# %%


# %%

# data_test = (NiftiNiftiList.from_folder(p,extensions={".gz"},recurse=True)
#       .split_by_folder(train='TRAIN', valid='VALID')
#       .label_from_func(get_y_fn)
#       .databunch(bs=1))
px = Path('/tsi/clusterhome/glabarbera/acgan/Database_GAN/trainA')
py = Path('/tsi/clusterhome/glabarbera/acgan/Database_GAN/trainB_new')

data_testx = (NumpyNumpyList.from_folder(px, extensions={".npz"}, recurse=True)
             .split_none()
             .label_empty()
             .databunch(bs=1, device=device))
print(data_testx)
# %%
final = []
from matplotlib import pyplot as plt
fig = plt.figure()
for j in range(60):

    a,_ = data_testx.one_batch()
    # %%
    print(a.shape)
    im = a.squeeze(1)
    print(im.shape)
    print(im.max())
    print(im.min())

    # %%

    # %%
    #for i in range(len(px)):
    score = np.zeros((im.shape[1]))
    for i in range(0, im.shape[1]):
        with torch.no_grad():
            sc = net(im[:,i:i+1,:, :].cuda())
        score[i] = sc.cpu().detach().numpy()
    if j==0:
        plt.plot(np.linspace(0,1,im.shape[1]),score, color="blue", label='ceCT')
    else:
        plt.plot(np.linspace(0,1,im.shape[1]),score, color="blue")

# %%

# VITAMIN.OrthoViewer(final.squeeze())

# %%

# VITAMIN.OrthoViewer(im_mr.squeeze().numpy())

# %%
#plt.show()
#fig.savefig('./testA', bbox_inches='tight')
data_testy = (NumpyNumpyList.from_folder(py, extensions={".npz"}, recurse=True)
             .split_none()
             .label_empty()
             .databunch(bs=1, device=device))
print(data_testy)
for j in range(60):
    b,_ = data_testy.one_batch()

    # %%

    im1 = b.squeeze(1)
    print(im1.shape)
    print(im1.max())
    print(im1.min())

    # %%

    score1 = np.zeros((im1.shape[1]))
    for i in range(0, im1.shape[1]):
        with torch.no_grad():
            sc = net(im1[:,i:i+1,:, :].cuda())
        score1[i] = sc.cpu().detach().numpy()
    if j==0:
        plt.plot(np.linspace(0,1,im1.shape[1]),score1, color="red", label='CT')
    else:
        plt.plot(np.linspace(0,1,im1.shape[1]),score1, color="red")
#plt.ylim((-100, 100))
plt.legend()
plt.show()
fig.savefig('./test', bbox_inches='tight')
#print(final1.shape)

#print(np.mean(np.abs(final[:min(im.shape[1], im1.shape[1])] - final1[:min(im.shape[1], im1.shape[1])])))

from random import randint
from PIL import Image
def np_to_img(I,massimo = None, minimo = None):
    if massimo == None:
        massimo = I.max()
    if minimo == None:
        minimo = I.min()
    I8 = (((I - minimo) / (massimo - minimo)) * 255.0).astype(np.uint8)
    img = Image.fromarray(I8)

    return img
ncols = 2  # number of columns in final grid of images
nrows = 5  # looking at all images takes some time
f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
for axis in axes.flatten():
    axis.set_axis_off()
    axis.set_aspect('equal')

q = 0
for i in range(0,im.shape[1],im.shape[1]//4):
    print(i)
    with torch.no_grad():
        sc = net(im[:,i:i+1,:, :].cuda())
    sc = sc.cpu().detach().numpy().flatten()
    print(sc)
    axes[q,0].set_title('A - slice ' + str(i)+': '+ str(sc), fontsize= 20)
    axes[q,0].imshow(np_to_img(im[0,i, :, :].data.cpu().numpy()), cmap="gray")
    q += 1

q = 0
for i in range(0,im1.shape[1],im1.shape[1]//4):
    print(i)
    with torch.no_grad():
        sc1 = net(im1[:,i:i+1,:, :].cuda())
    sc1 = sc1.cpu().detach().numpy().flatten()
    print(sc1)
    axes[q,1].set_title('B - slice '+ str(i)+': '+str(sc1), fontsize= 20)
    axes[q,1].imshow(np_to_img(im1[0,i,:, :].data.cpu().numpy()), cmap="gray")
    q += 1
f.show()
f.savefig('./examplesAB', bbox_inches='tight')



ncols = 3  # number of columns in final grid of images
nrows = 1  # looking at all images takes some time
#diff = len(score1) - len(score)
f, axes = plt.subplots(nrows, ncols)
#maxx = max(im1.shape[1],im.shape[1])
print(im1.shape)
axes[0].imshow(np_to_img(im1[0,:,im1.shape[2]//2, :].data.cpu().numpy()), cmap="gray")
axes[0].set_ylim(0,im1.shape[1])
print(im.shape)
axes[1].imshow(np_to_img(im[0,:,im.shape[2]//2, :].data.cpu().numpy()), cmap="gray")
axes[1].set_ylim(0,im.shape[1])
axes[2].plot(score1[:],np.linspace(0,1,len(score1)))
axes[2].plot(score[:],np.linspace(0,1,len(score)))
f.show()
f.savefig('./examples', bbox_inches='tight')

'''

import os

p2 = './Test-Set'
(_, _, filenames_imts) = next(os.walk(p2))
filenames_imts = sorted(filenames_imts)
ncols = 2  # number of columns in final grid of images
nrows = 4  # looking at all images takes some time
f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
for axis in axes.flatten():
    axis.set_axis_off()
    axis.set_aspect('equal')
q = 0
for i in range(len(filenames_imts)):
    x =np.array(Image.open(os.path.join(p2, filenames_imts[i])))
    print(filenames_imts[i])
    print(x.shape)
    print(x.max())
    print(x.min())
    x = (x/255.0) * 2.0 - 1.0
    # x = sitk.GetArrayFromImage(sitk.ReadImage(str(fn), sitk.sitkFloat32))
    im2 = torch.Tensor(x[np.newaxis,np.newaxis, ...])
    with torch.no_grad():
        sc = net(im2.cuda())
    sc = sc.cpu().detach().numpy().flatten()
    print(sc)
    if i<4:
        axes[q,0].set_title('input: '+ str(sc), fontsize= 20)
        axes[q,0].imshow(np_to_img(im2[0, 0, :, :].data.cpu().numpy()), cmap="gray")
        q += 1
    elif i==4:
        q = 0
        axes[q,1].set_title('output: '+ str(sc), fontsize= 20)
        axes[q,1].imshow(np_to_img(im2[0, 0, :, :].data.cpu().numpy()), cmap="gray")
        q += 1
    else:
        axes[q,1].set_title('output: '+ str(sc), fontsize= 20)
        axes[q,1].imshow(np_to_img(im2[0, 0, :, :].data.cpu().numpy()), cmap="gray")
        q += 1
f.show()
f.savefig('./examples2')

'''
