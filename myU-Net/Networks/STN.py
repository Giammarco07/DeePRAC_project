import torch.nn as nn
import torch.nn.functional as F
import torch

def stn_mode(x, theta, mode='affine', padding = 'border'):
    if mode == 'affine':
        theta1 = theta.view(-1, 2, 3)
    else:
        theta1 = torch.zeros((x.size()[0], 2, 3), dtype=torch.float32, requires_grad=True).to(torch.device("cuda"))
        '''
        PyTorch or more precisely autograd is not very good in handling in-place operations, 
        especially on those tensors with the requires_grad flag set to True.
        Generally you should avoid in-place operations where it is possible, in some cases it can work, 
        but you should always avoid in-place operations on tensors where you set requires_grad to True.
        Unfortunately there are not many pytorch functions to help out on this problem. 
        So you would have to use a helper tensor to avoid the in-place operation.
        '''
        theta1 = theta1 + 0  # new tensor, out-of-place operation (get around the problem, it is like "magic!")

        theta1[:, 0, 0] = 1.0
        theta1[:, 1, 1] = 1.0
        if mode == 'translation':
            #theta = 2
            theta1[:, 0, 2] = F.tanh(theta[:, 0])
            theta1[:, 1, 2] = F.tanh(theta[:, 1])
        elif mode == 'rotation':
            #theta = 1
            angle = theta[:, 0]
            theta1[:, 0, 0] = torch.cos(angle)
            theta1[:, 0, 1] = -torch.sin(angle)
            theta1[:, 1, 0] = torch.sin(angle)
            theta1[:, 1, 1] = torch.cos(angle)
        elif mode == 'scale':
            # theta = 2
            theta1[:, 0, 0] = F.relu(theta[:, 0]) + 1e-20
            theta1[:, 1, 1] = F.relu(theta[:, 1]) + 1e-20
        elif mode == 'shear':
            # theta = 2
            theta1[:, 0, 1] = theta[:, 0]
            theta1[:, 1, 0] = theta[:, 1]
        elif mode == 'rotation_scale':
            # theta = 3
            angle = theta[:, 0]
            sx = F.relu(theta[:, 1]) + 1e-20
            sy = F.relu(theta[:, 2]) + 1e-20
            theta1[:, 0, 0] = torch.cos(angle) * sx
            theta1[:, 0, 1] = -torch.sin(angle) * sx
            theta1[:, 1, 0] = torch.sin(angle) * sy
            theta1[:, 1, 1] = torch.cos(angle) * sy
        elif mode == 'translation_scale':
            # theta = 4
            theta1[:, 0, 2] = F.tanh(theta[:, 0])
            theta1[:, 1, 2] = F.tanh(theta[:, 1])
            theta1[:, 0, 0] = F.relu(theta[:, 2]) + 1e-20
            theta1[:, 1, 1] = F.relu(theta[:, 3]) + 1e-20
        elif mode == 'rotation_translation':
            # theta = 3
            angle = theta[:, 0]
            theta1[:, 0, 0] = torch.cos(angle)
            theta1[:, 0, 1] = -torch.sin(angle)
            theta1[:, 1, 0] = torch.sin(angle)
            theta1[:, 1, 1] = torch.cos(angle)
            theta1[:, 0, 2] = F.tanh(theta[:, 1])
            theta1[:, 1, 2] = F.tanh(theta[:, 2])
        elif mode == 'rotation_translation_scale':
            # theta = 5
            angle = theta[:, 0]
            #sx = F.relu(theta[:, 3]) + 1e-20
            #sy = F.relu(theta[:, 4]) + 1e-20
            sx = F.hardtanh(theta[:, 3], min_val=0.5, max_val=2.0)
            sy = F.hardtanh(theta[:, 4], min_val=0.5, max_val=2.0)
            theta1[:, 0, 0] = torch.cos(angle) * sx
            theta1[:, 0, 1] = -torch.sin(angle) * sx
            theta1[:, 1, 0] = torch.sin(angle) * sy
            theta1[:, 1, 1] = torch.cos(angle) * sy
            theta1[:, 0, 2] = F.tanh(theta[:, 1])
            theta1[:, 1, 2] = F.tanh(theta[:, 2])


    grid = F.affine_grid(theta1, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode=padding)

    return x, theta1

class net(nn.Module):
    def __init__(self, ngpu,channel_dim,patch_size, mode='affine'):
        super(net, self).__init__()
        self.ngpu = ngpu
        self.mode = mode
        if mode=='affine':
            channel_out=6
        elif mode == 'rotation':
            channel_out=1
        elif mode == 'translation' or mode == 'scale' or mode=='shear':
            channel_out=2
        elif mode == 'rotation_scale' or  mode == 'rotation_translation':
            channel_out=3
        elif mode == 'translation_scale':
            channel_out=4
        elif mode == 'rotation_translation_scale':
            channel_out=5

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(channel_dim, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * int((patch_size[0]/4)-4) * int((patch_size[1]/4)-4), 1024),
            nn.ReLU(True),
            nn.Linear(1024, 32),
            nn.ReLU(True),
            nn.Linear(32, channel_out)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[4].weight.data.zero_()
        if mode=='affine':
            self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif mode == 'rotation':
            self.fc_loc[4].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        elif mode == 'translation' or mode=='shear':
            self.fc_loc[4].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        elif mode == 'scale':
            self.fc_loc[4].bias.data.copy_(torch.tensor([1, 1], dtype=torch.float))
        elif mode == 'rotation_scale':
            self.fc_loc[4].bias.data.copy_(torch.tensor([0, 1, 1], dtype=torch.float))
        elif  mode == 'rotation_translation':
            self.fc_loc[4].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
        elif mode == 'translation_scale':
            self.fc_loc[4].bias.data.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.float))
        elif mode == 'rotation_translation_scale':
            self.fc_loc[4].bias.data.copy_(torch.tensor([0, 0, 0, 1, 1], dtype=torch.float))

        # Spatial transformer network forward function
    def forward(self, x, padding='border'):
        mode = self.mode
        xs = self.localization(x)
        xs = xs.view(-1, 10 * int((x.size()[2] / 4) - 4) * int((x.size()[3] / 4) - 4))
        theta = self.fc_loc(xs)
        output,theta_final = stn_mode(x,theta,mode=mode,padding = padding)

        return output, theta_final