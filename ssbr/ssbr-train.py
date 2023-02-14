from functools import singledispatch

from fastai_function import slice_sampler,slice_sampler_normalized, NumpyNumpyList, NumpyImageList
from fastai.vision import *
from fastai.basics import *

from matplotlib import pyplot as plt
from PIL import Image
def np_to_img(I,massimo = None, minimo = None):
    if massimo == None:
        massimo = I.max()
    if minimo == None:
        minimo = I.min()
    I8 = (((I - minimo) / (massimo - minimo)) * 255.0).astype(np.uint8)
    img = Image.fromarray(I8)

    return img

def keep_mask(image):
    mask = image.clone()
    white = torch.ones((1, image.size()[-1], image.size()[-1]), requires_grad=False,
                       dtype=torch.float).to("cuda:0")
    black = torch.zeros((1, image.size()[-1], image.size()[-1]), requires_grad=False,
                        dtype=torch.float).to("cuda:0")
    for i in range(mask.size()[0]):
        mask[i] = torch.where(mask[i] >= (mask[i].min() + 0.1), torch.where(mask[i] < (mask[i].min() + 0.1), mask[i], white),
                           black)
    return mask


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
# %%
#path = 'E:/Francesco'
#path = '/ldaphome/glabarbera/UNIT/datasets'
p = Path('/tsi/clusterhome/glabarbera/acgan/Database_GAN/trainA')
p1 = Path('/tsi/clusterhome/glabarbera/acgan/Database_GAN/trainB_new')

src = (ItemList.from_folder(p1, extensions={".npz"}, recurse=True))
print(src)
# %%
tfms = [slice_sampler_normalized(num_slices=8, equidistance_range=(1, 10))]

data = (NumpyNumpyList.from_folder(p, extensions={".npz"}, recurse=True)
        .add(src)
        .split_by_rand_pct()
        .label_empty()
        .transform((tfms, tfms), tfm_y=False)
        .databunch(bs=32,device=device))
print(data)
# %%
x = data.one_batch()[0]
print(x.get_device())
print(x.shape)
print(x.max())
print(x.min())

fig = plt.figure()
plt.imshow(np_to_img(x[0,x.shape[1]//2,:, :].data.cpu().numpy()), cmap="gray")
fig.savefig('./prova', bbox_inches='tight')
fig = plt.figure()
plt.imshow(np_to_img(x[-1,x.shape[1]//2,:, :].data.cpu().numpy()), cmap="gray")
fig.savefig('./provabis', bbox_inches='tight')

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
        init = x.clone()
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, 1,  x.size(2), x.size(3)).repeat(1, 3, 1, 1)
        x = self.module(x)
        x = self.lin(x)
        #x = self.output(self.lin(x))
        #x = x.view(t * n,1,-1)
        #x = self.linear(x)
        x = x.mean([2, 3])
        x = x.view(t, n, -1)
        return x,init

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


# %%

m = SequenceWise(models.resnet34(pretrained=False)).to(device)
print(m)

# %%

class L1LossFlat(nn.SmoothL1Loss):
    def forward(self, input: Tensor, target: Tensor) -> Rank0Tensor:
        return super().forward(input.reshape(-1), target.float().reshape(-1))


smoothL1 = L1LossFlat().to(device)

def loss_order(scores_true, scores_predd):
    """
    Implements a ordering loss for slice scores.
    Scores is a tensor of dimension B x S where B is the number of volumes in a batch and S is the number of slice
    scores per volume. The score should have ascending order

    :param scores_pred: Not used
    :param scores_pred: Tensor of dimension B x S, where B is the number of volumes and S the number of slice scores.
    :type scores_pred:
    :return: order loss tensor
    """
    eps = 1e-10
    scores_pred = scores_predd[0]
    score_diff = scores_pred[:, 1:] - scores_pred[:, 0:-1]
    loss = -torch.sum(torch.log(torch.sigmoid(100*score_diff.flatten()) + eps))
    return loss


def loss_distance(scores_true, scores_predd):
    """Implements a distance loss between slice scores.
    Scores is a tensor of dimension B x S. Each slices should have equidistant scores, since all input images should be
    equidistant.

    :param scores_true: Not used
    :param scores_pred: Tensor of dimension B x S, where B is the number of volumes and S the number of slice scores.
    :type scores_pred:
    :return: distance loss tensor
    """
    scores_pred = scores_predd[0]
    score_diff = scores_pred[:, 1:] - scores_pred[:, 0:-1]
    loss = torch.sum(smoothL1(score_diff[:, 1:], score_diff[:, 0:-1]))
    return loss

def loss_difference(scores_true, scores_predd):
    """Implements a distance loss between slice scores.
    Scores is a tensor of dimension B x S. Each slices should have equidistant scores, since all input images should be
    equidistant.

    :param scores_true: Not used
    :param scores_pred: Tensor of dimension B x S, where B is the number of volumes and S the number of slice scores.
    :type scores_pred:
    :return: distance loss tensor
    """
    scores_pred = scores_predd[0]
    data = scores_predd[1]
    masks = keep_mask(data)
    scores_true = (1 - (torch.sum(masks[:, 1:]*masks[:, 0:-1],dim=(2,3),keepdim=True))/torch.sum(masks[:, 1:],dim=(2,3),keepdim=True))[...,0]
    scores_true += ((2 - torch.sum(scores_true,dim=(1),keepdim=True))/7)
    scores_diff = scores_pred[:, 1:] - scores_pred[:, 0:-1]
    loss = torch.sum(smoothL1(scores_diff, scores_true))
    return loss


def loss_normalization(scores_true, scores_predd):
    """Implements a distance loss between slice scores.
    Scores is a tensor of dimension B x S. Each slices should have equidistant scores, since all input images should be
    equidistant.

    :param scores_true: Not used
    :param scores_pred: Tensor of dimension B x S, where B is the number of volumes and S the number of slice scores.
    :type scores_pred:
    :return: distance loss tensor
    """
    scores_pred = scores_predd[0]
    print(scores_pred[0],scores_pred[-1])
    score_begin = torch.sum(smoothL1(scores_pred[:, 0], -1 * torch.ones_like(scores_pred[:, 0])))
    score_end = torch.sum(smoothL1(scores_pred[:, -1], 1 * torch.ones_like(scores_pred[:, -1])))
    loss = score_begin + score_end
    return loss


class SsbrLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma = 0, delta=0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, input, target, **kwargs):
        a = loss_distance(target, input)  
        b = loss_order(target,input)
        c = loss_normalization(target,input)
        d = loss_difference(target,input)
        print(b, c, d)
        return self.alpha * a + self.beta * b + self.gamma * c + self.delta * d


# %%
def addSaveCallbackClassifier(learner):
    #Add a save model callback
    learner.callback_fns.append(partial(callbacks.SaveModelCallback, monitor='valid_loss', name='classifier-model-saved-by-callback'))


my_loss = SsbrLoss(alpha=0, beta=0.005, gamma = 10,delta=10).to(device)
learn = Learner(data, m, loss_func=my_loss,path='.')


learn.unfreeze()
addSaveCallbackClassifier(learn)
# %%

#learn.lr_find()
#learn.recorder.plot(suggestion=True)

# %%;
learn.fit(500, lr=5e-2)
learn.fit(250, lr=1e-2)
learn.fit(250, lr=5e-3)

# %%





# %%



# %%
fig = learn.recorder.plot_losses(return_fig=True)
fig.savefig('./plot', bbox_inches='tight')

# %%
