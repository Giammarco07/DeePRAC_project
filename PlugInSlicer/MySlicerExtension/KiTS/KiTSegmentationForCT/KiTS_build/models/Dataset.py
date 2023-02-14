import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets

def rescaled(data,preprocessing):
    np.clip(data, preprocessing[3], preprocessing[2], out=data)
    data = (data - preprocessing[0]) / preprocessing[1]   # data - mean / std_dev of foreground

    return data

class CustomDataset_new(Dataset):
    def __init__(self, patch_ids,imaget,dpatch_size,hpatch_size,wpatch_size, preprocessing):  # initial logic happens like transform
        self.patch_ids = patch_ids
        self.imaget = imaget
        self.dpatch_size = dpatch_size
        self.hpatch_size = hpatch_size
        self.wpatch_size = wpatch_size
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        (d, h, w) = self.patch_ids[index]
        image = self.imaget[d:d + self.dpatch_size, h:h + self.hpatch_size, w:w + self.wpatch_size]
        image = rescaled(image, self.preprocessing)
        t_image = torch.from_numpy(image).type('torch.FloatTensor')

        return t_image.unsqueeze(0)

    def __len__(self):  # return count of sample we have
        return len(self.patch_ids)


def Prepare_Test_Data_new(patch_ids,imaget,dpatch_size,hpatch_size,wpatch_size,batch_size,workers, preprocessing):
    test_dataset = CustomDataset_new(patch_ids,imaget,dpatch_size,hpatch_size,wpatch_size, preprocessing)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=workers)

    return test_loader