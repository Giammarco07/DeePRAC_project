import torch.nn as nn
import torch.nn.functional as F
import torch

class net(nn.Module):
    def __init__(self, ngpu,channel_dim, patch_size):
        super(net, self).__init__()
        self.ngpu = ngpu

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
            nn.Linear(10 * int((patch_size[0] / 4) - 4) * int((patch_size[1] / 4) - 4), 1024),
            nn.ReLU(True),
            nn.Linear(1024, 32),
            nn.ReLU(True),
            nn.Linear(32, 4),
            nn.ReLU(True)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([patch_size[0]/4, patch_size[0]*3/4, patch_size[1]/4, patch_size[1]*3/4], dtype=torch.float))


        # Spatial transformer network forward function
    def forward(self, im_tensor, mode = 'normal'):
        xs = self.localization(im_tensor)
        xs = xs.view(-1, 10 * int((im_tensor.size()[2] / 4) - 4) * int((im_tensor.size()[3] / 4) - 4))
        sz_crop = self.fc_loc(xs) #regressor for 2*2 matrix
        theta_crop = torch.zeros((im_tensor.size()[0],2,3),requires_grad=True, dtype=torch.float).to(torch.device("cuda"))
        '''
                PyTorch or more precisely autograd is not very good in handling in-place operations, 
                especially on those tensors with the requires_grad flag set to True.
                Generally you should avoid in-place operations where it is possible, in some cases it can work, 
                but you should always avoid in-place operations on tensors where you set requires_grad to True.
                Unfortunately there are not many pytorch functions to help out on this problem. 
                So you would have to use a helper tensor to avoid the in-place operation.
        '''
        theta_crop = theta_crop + 0 # new tensor, out-of-place operation (get around the problem)

        if mode == 'normal':
            for i in range(im_tensor.size()[0]):
                theta_crop[i,0,0] = ((sz_crop[i, 1] - sz_crop[i, 0])/(im_tensor.size()[2]*1.0)) + 1e-20 #x2-x1/w
                theta_crop[i,0,2] = ((sz_crop[i, 1] + sz_crop[i, 0])/(im_tensor.size()[2]*1.0)) - 1 #x2+x1/w - 1
                theta_crop[i,1,1] = ((sz_crop[i, 3] - sz_crop[i, 2])/(im_tensor.size()[3]*1.0))  + 1e-20 #y2-y1/h
                theta_crop[i,1,2] = ((sz_crop[i, 3] + sz_crop[i, 2])/(im_tensor.size()[3]*1.0)) - 1 #y2+y1/h -1
        elif mode == 'security':
            for i in range(im_tensor.size()[0]):
                # adding safety margins of 0.2 of the size
                addx = int(im_tensor.size()[2] * 0.2)
                addy = int(im_tensor.size()[3] * 0.2)
                if (sz_crop[i, 1] - sz_crop[i, 0]) < (im_tensor.size()[2] - addx):
                    theta_crop[i, 0, 0] = ((sz_crop[i, 1] - sz_crop[i, 0] + addx) / (
                            im_tensor.size()[2] * 1.0)) + 1e-20  # x2-x1/w
                    if sz_crop[i, 0] < (addx / 2):
                        theta_crop[i, 0, 2] = ((sz_crop[i, 1] + addx - sz_crop[i, 0]) / (
                                im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    elif sz_crop[i, 1] > (im_tensor.size()[2] - (addx / 2)):
                        theta_crop[i, 0, 2] = ((im_tensor.size()[2] + sz_crop[i, 0] - addx + im_tensor.size()[2] -
                                                sz_crop[i, 1]) / (
                                                       im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    else:
                        theta_crop[i, 0, 2] = ((sz_crop[i, 1] + sz_crop[i, 0]) / (
                                im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                else:
                    theta_crop[i, 0, 0] = ((sz_crop[i, 1] - sz_crop[i, 0]) / (
                            im_tensor.size()[2] * 1.0)) + 1e-20  # x2-x1/w
                    theta_crop[i, 0, 2] = ((sz_crop[i, 1] + sz_crop[i, 0]) / (
                            im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                if (sz_crop[i, 3] - sz_crop[i, 2]) < (im_tensor.size()[3] - addy):
                    theta_crop[i, 1, 1] = ((sz_crop[i, 3] - sz_crop[i, 2] + addy) / (
                            im_tensor.size()[3] * 1.0)) + 1e-20  # y2-y1/h
                    if sz_crop[i, 2] < (addy / 2):
                        theta_crop[i, 1, 2] = ((sz_crop[i, 3] + addy - sz_crop[i, 2]) / (
                                im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h - 1
                    elif sz_crop[i, 3] > (im_tensor.size()[3] - (addy / 2)):
                        theta_crop[i, 1, 2] = ((im_tensor.size()[3] + sz_crop[i, 2] - addy + im_tensor.size()[3] -
                                                sz_crop[
                                                    i, 3]) / (im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h- 1
                    else:
                        theta_crop[i, 1, 2] = ((sz_crop[i, 3] + sz_crop[i, 2]) / (
                                im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h - 1
                else:
                    theta_crop[i, 1, 1] = ((sz_crop[i, 3] - sz_crop[i, 2]) / (
                            im_tensor.size()[3] * 1.0)) + 1e-20  # y2-y1/h
                    theta_crop[i, 1, 2] = ((sz_crop[i, 3] + sz_crop[i, 2]) / (
                            im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h - 1

        grid_cropped = F.affine_grid(theta_crop, im_tensor.size(), align_corners=False)
        im_tensor_cropped = F.grid_sample(im_tensor, grid_cropped, align_corners=False, mode='bilinear') #, padding_mode="border"

        return im_tensor_cropped, theta_crop

class net_new(nn.Module):
    def __init__(self, ngpu,channel_dim, patch_size, use_bias=True):
        super(net_new, self).__init__()
        self.ngpu = ngpu

        # Spatial transformer localization-network
        self.loc1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.loc2 = nn.Sequential(
            nn.Conv2d(9, 8, 3, stride=(2, 2), padding=1, bias=use_bias),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )

        self.loc3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.loc4 = nn.Sequential(
            nn.Conv2d(24, 16, 3, stride=(2, 2), padding=1, bias=use_bias),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.loc5 = nn.Sequential(
            nn.Conv2d(16, 24, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(24),
            nn.ReLU(True)
        )
        self.loc6 = nn.Sequential(
            nn.Conv2d(40, 24, 3, stride=(2, 2), padding=1, bias=use_bias),
            nn.BatchNorm2d(24),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(24 * int(patch_size[0] / 8) * int(patch_size[1] / 8), 1024),
            nn.ReLU(True),
            nn.Linear(1024, 32),
            nn.ReLU(True),
            nn.Linear(32, 4)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([1, 1, 0, 0], dtype=torch.float))


        # Spatial transformer network forward function
    def forward(self, im_tensor, mode = 'normal'):
        x1 = self.loc1(im_tensor)
        x2 = self.loc2(torch.cat((x1, im_tensor),1))
        x3 = self.loc3(x2)
        x4 = self.loc4(torch.cat((x3, x2),1))
        x5 = self.loc5(x4)
        xs = self.loc6(torch.cat((x5, x4),1))

        xs = xs.view(-1, 24 * int(im_tensor.size()[2] / 8)* int(im_tensor.size()[3] / 8))
        sz_crop = self.fc_loc(xs) #regressor for 2*2 matrix
        theta_crop = torch.zeros((im_tensor.size()[0],2,3),requires_grad=True, dtype=torch.float).to(torch.device("cuda"))
        theta_crop = theta_crop + 0
        if mode == 'normal':
            for i in range(im_tensor.size()[0]):
                theta_crop[i,0,0] = F.sigmoid(sz_crop[i, 0]) #x2-x1/w
                theta_crop[i,0,2] = F.tanh(sz_crop[i, 2]) #x2+x1/w - 1
                theta_crop[i,1,1] = F.sigmoid(sz_crop[i, 1]) #y2-y1/h
                theta_crop[i,1,2] = F.tanh(sz_crop[i, 3]) #y2+y1/h -1
        elif mode == 'security':
            for i in range(im_tensor.size()[0]):
                # adding safety margins of 0.2 of the size
                addx = int(im_tensor.size()[2] * 0.2)
                addy = int(im_tensor.size()[3] * 0.2)
                if (sz_crop[i, 1] - sz_crop[i, 0]) < (im_tensor.size()[2] - addx):
                    theta_crop[i, 0, 0] = ((sz_crop[i, 1] - sz_crop[i, 0] + addx) / (
                            im_tensor.size()[2] * 1.0)) + 1e-20  # x2-x1/w
                    if sz_crop[i, 0] < (addx / 2):
                        theta_crop[i, 0, 2] = ((sz_crop[i, 1] + addx - sz_crop[i, 0]) / (
                                im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    elif sz_crop[i, 1] > (im_tensor.size()[2] - (addx / 2)):
                        theta_crop[i, 0, 2] = ((im_tensor.size()[2] + sz_crop[i, 0] - addx + im_tensor.size()[2] -
                                                sz_crop[i, 1]) / (
                                                       im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    else:
                        theta_crop[i, 0, 2] = ((sz_crop[i, 1] + sz_crop[i, 0]) / (
                                im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                else:
                    theta_crop[i, 0, 0] = ((sz_crop[i, 1] - sz_crop[i, 0]) / (
                            im_tensor.size()[2] * 1.0)) + 1e-20  # x2-x1/w
                    theta_crop[i, 0, 2] = ((sz_crop[i, 1] + sz_crop[i, 0]) / (
                            im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                if (sz_crop[i, 3] - sz_crop[i, 2]) < (im_tensor.size()[3] - addy):
                    theta_crop[i, 1, 1] = ((sz_crop[i, 3] - sz_crop[i, 2] + addy) / (
                            im_tensor.size()[3] * 1.0)) + 1e-20  # y2-y1/h
                    if sz_crop[i, 2] < (addy / 2):
                        theta_crop[i, 1, 2] = ((sz_crop[i, 3] + addy - sz_crop[i, 2]) / (
                                im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h - 1
                    elif sz_crop[i, 3] > (im_tensor.size()[3] - (addy / 2)):
                        theta_crop[i, 1, 2] = ((im_tensor.size()[3] + sz_crop[i, 2] - addy + im_tensor.size()[3] -
                                                sz_crop[
                                                    i, 3]) / (im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h- 1
                    else:
                        theta_crop[i, 1, 2] = ((sz_crop[i, 3] + sz_crop[i, 2]) / (
                                im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h - 1
                else:
                    theta_crop[i, 1, 1] = ((sz_crop[i, 3] - sz_crop[i, 2]) / (
                            im_tensor.size()[3] * 1.0)) + 1e-20  # y2-y1/h
                    theta_crop[i, 1, 2] = ((sz_crop[i, 3] + sz_crop[i, 2]) / (
                            im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h - 1

        grid_cropped = F.affine_grid(theta_crop, im_tensor.size(), align_corners=False)
        im_tensor_cropped = F.grid_sample(im_tensor, grid_cropped, align_corners=False, mode='bilinear') #, padding_mode="border"

        return im_tensor_cropped, theta_crop


class net_2(nn.Module):
    def __init__(self, ngpu,channel_dim, patch_size):
        super(net_2, self).__init__()
        self.ngpu = ngpu

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
            nn.Linear(10 * int((patch_size[0] / 4) - 4) * int((patch_size[1] / 4) - 4), 1024),
            nn.ReLU(True),
            nn.Linear(1024, 32),
            nn.ReLU(True),
            nn.Linear(32, 4),
            nn.Hardtanh(min_val = 0.0, max_val = (patch_size[0]-1), inplace=True)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([0.0, patch_size[0]/2.0, 0.0, patch_size[1]/2.0], dtype=torch.float))


        # Spatial transformer network forward function
    def forward(self, im_tensor, mode = 'normal'):
        xs = self.localization(im_tensor)
        xs = xs.view(-1, 10 * int((im_tensor.size()[2] / 4) - 4) * int((im_tensor.size()[3] / 4) - 4))
        sz_crop = self.fc_loc(xs) #regressor for 2*2 matrix
        theta_crop = torch.zeros((im_tensor.size()[0],2,3),requires_grad=True, dtype=torch.float).to(torch.device("cuda"))
        theta_crop = theta_crop + 0
        img_crop = torch.zeros((im_tensor.size()),requires_grad=True, dtype=torch.float).to(torch.device("cuda"))
        img_crop = img_crop + im_tensor.min()

        if mode == 'normal':
            theta_crop[:,0,0] = ((sz_crop[:, 1] - sz_crop[:, 0])/(im_tensor.size()[2]*1.0)) + 1e-20   #x2-x1/w
            theta_crop[:,0,2] = ((sz_crop[:, 1] + sz_crop[:, 0])/(im_tensor.size()[2]*1.0)) - 1 #x2+x1/w - 1
            theta_crop[:,1,1] = ((sz_crop[:, 3] - sz_crop[:, 2])/(im_tensor.size()[3]*1.0)) + 1e-20 #y2-y1/h
            theta_crop[:,1,2] = ((sz_crop[:, 3] + sz_crop[:, 2])/(im_tensor.size()[3]*1.0)) - 1 #y2+y1/h -1
        elif mode == 'security':
            for i in range(im_tensor.size()[0]):
                # adding safety margins of 0.2 of the size
                addx = int(im_tensor.size()[2] * 0.2)
                addy = int(im_tensor.size()[3] * 0.2)
                if (sz_crop[i, 1] - sz_crop[i, 0]) < (im_tensor.size()[2] - addx):
                    theta_crop[i, 0, 0] = ((sz_crop[i, 1] - sz_crop[i, 0] + addx) / (
                            im_tensor.size()[2] * 1.0)) + 1e-20  # x2-x1/w
                    if sz_crop[i, 0] < (addx / 2):
                        theta_crop[i, 0, 2] = ((sz_crop[i, 1] + addx - sz_crop[i, 0]) / (
                                im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    elif sz_crop[i, 1] > (im_tensor.size()[2] - (addx / 2)):
                        theta_crop[i, 0, 2] = ((im_tensor.size()[2] + sz_crop[i, 0] - addx + im_tensor.size()[2] -
                                                sz_crop[i, 1]) / (
                                                       im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    else:
                        theta_crop[i, 0, 2] = ((sz_crop[i, 1] + sz_crop[i, 0]) / (
                                im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                else:
                    theta_crop[i, 0, 0] = ((sz_crop[i, 1] - sz_crop[i, 0]) / (
                            im_tensor.size()[2] * 1.0)) + 1e-20  # x2-x1/w
                    theta_crop[i, 0, 2] = ((sz_crop[i, 1] + sz_crop[i, 0]) / (
                            im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                if (sz_crop[i, 3] - sz_crop[i, 2]) < (im_tensor.size()[3] - addy):
                    theta_crop[i, 1, 1] = ((sz_crop[i, 3] - sz_crop[i, 2] + addy) / (
                            im_tensor.size()[3] * 1.0)) + 1e-20  # y2-y1/h
                    if sz_crop[i, 2] < (addy / 2):
                        theta_crop[i, 1, 2] = ((sz_crop[i, 3] + addy - sz_crop[i, 2]) / (
                                im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h - 1
                    elif sz_crop[i, 3] > (im_tensor.size()[3] - (addy / 2)):
                        theta_crop[i, 1, 2] = ((im_tensor.size()[3] + sz_crop[i, 2] - addy + im_tensor.size()[3] -
                                                sz_crop[
                                                    i, 3]) / (im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h- 1
                    else:
                        theta_crop[i, 1, 2] = ((sz_crop[i, 3] + sz_crop[i, 2]) / (
                                im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h - 1
                else:
                    theta_crop[i, 1, 1] = ((sz_crop[i, 3] - sz_crop[i, 2]) / (
                            im_tensor.size()[3] * 1.0)) + 1e-20  # y2-y1/h
                    theta_crop[i, 1, 2] = ((sz_crop[i, 3] + sz_crop[i, 2]) / (
                            im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h - 1
        for i in range(im_tensor.size()[0]):
            grid_cropped = F.affine_grid(theta_crop[i].unsqueeze(0), (1, 1, int(sz_crop[i, 3]) - int(sz_crop[i, 2]) + 1, int(sz_crop[i, 1]) - int(sz_crop[i, 0]) + 1), align_corners=False)
            im_tensor_cropped = F.grid_sample(im_tensor[i].unsqueeze(0), grid_cropped, align_corners=False, mode='bilinear') #, padding_mode="border"
            img_crop[i,0,int(sz_crop[i, 2]):int(sz_crop[i, 3])+1,int(sz_crop[i, 0]):int(sz_crop[i, 1])+1] = im_tensor_cropped

        return img_crop, theta_crop


class net_3(nn.Module):
    def __init__(self, ngpu, channel_dim, patch_size, use_bias = True):
        super(net_3, self).__init__()
        self.ngpu = ngpu

        # Spatial transformer localization-network
        self.loc1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.loc2 = nn.Sequential(
            nn.Conv2d(9, 8, 3, stride=(2, 2), padding=1, bias=use_bias),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )

        self.loc3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.loc4 = nn.Sequential(
            nn.Conv2d(24, 16, 3, stride=(2, 2), padding=1, bias=use_bias),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.loc5 = nn.Sequential(
            nn.Conv2d(16, 24, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(24),
            nn.ReLU(True)
        )
        self.loc6 = nn.Sequential(
            nn.Conv2d(40, 24, 3, stride=(2, 2), padding=1, bias=use_bias),
            nn.BatchNorm2d(24),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(24 * int(patch_size[0] / 8) * int(patch_size[1] / 8), 1024),
            nn.ReLU(True),
            nn.Linear(1024, 4),
            nn.Hardtanh(min_val=0.0, max_val=(patch_size[0] - 1), inplace=True)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([patch_size[0]/4, patch_size[0]*3/4, patch_size[1]/4, patch_size[1]*3/4], dtype=torch.float))

        # Spatial transformer network forward function

    def forward(self, im_tensor, mode='normal'):
        x1 = self.loc1(im_tensor)
        x2 = self.loc2(torch.cat((x1, im_tensor),1))
        x3 = self.loc3(x2)
        x4 = self.loc4(torch.cat((x3, x2),1))
        x5 = self.loc5(x4)
        xs = self.loc6(torch.cat((x5, x4),1))

        xs = xs.view(-1, 24 * int(im_tensor.size()[2] / 8)* int(im_tensor.size()[3] / 8))
        sz_crop = self.fc_loc(xs)  # regressor for 2*2 matrix
        theta_crop = torch.zeros((im_tensor.size()[0], 2, 3), requires_grad=True, dtype=torch.float).to(
            torch.device("cuda"))
        theta_crop = theta_crop + 0
        img_crop = torch.zeros((im_tensor.size()), requires_grad=True, dtype=torch.float).to(torch.device("cuda"))
        img_crop = img_crop + im_tensor.min()

        if mode == 'normal':
            theta_crop[:, 0, 0] = ((sz_crop[:, 1] - sz_crop[:, 0]) / (im_tensor.size()[2] * 1.0)) + 1e-20  # x2-x1/w
            theta_crop[:, 0, 2] = ((sz_crop[:, 1] + sz_crop[:, 0]) / (im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
            theta_crop[:, 1, 1] = ((sz_crop[:, 3] - sz_crop[:, 2]) / (im_tensor.size()[3] * 1.0)) + 1e-20  # y2-y1/h
            theta_crop[:, 1, 2] = ((sz_crop[:, 3] + sz_crop[:, 2]) / (im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h -1
        elif mode == 'security':
            for i in range(im_tensor.size()[0]):
                # adding safety margins of 0.2 of the size
                addx = int(im_tensor.size()[2] * 0.2)
                addy = int(im_tensor.size()[3] * 0.2)
                if (sz_crop[i, 1] - sz_crop[i, 0]) < (im_tensor.size()[2] - addx):
                    if sz_crop[i, 0] < (addx / 2):
                        sz_crop[i, 1] += (addx / 2)
                    elif sz_crop[i, 1] > (im_tensor.size()[2] - (addx / 2)):
                        sz_crop[i, 0] -= (addx / 2)
                    else:
                        sz_crop[i, 1] += (addx / 2)
                        sz_crop[i, 0] -= (addx / 2)
                if (sz_crop[i, 3] - sz_crop[i, 2]) < (im_tensor.size()[3] - addy):
                    if sz_crop[i, 2] < (addy / 2):
                        sz_crop[i, 3] += (addy / 2)
                    elif sz_crop[i, 3] > (im_tensor.size()[3] - (addy / 2)):
                        sz_crop[i, 2] -= (addy / 2)
                    else:
                        sz_crop[i, 3] += (addy / 2)
                        sz_crop[i, 2] -= (addy / 2)

            theta_crop[:, 0, 0] = ((sz_crop[:, 1] - sz_crop[:, 0]) / (im_tensor.size()[2] * 1.0)) + 1e-20  # x2-x1/w
            theta_crop[:, 0, 2] = ((sz_crop[:, 1] + sz_crop[:, 0]) / (im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
            theta_crop[:, 1, 1] = ((sz_crop[:, 3] - sz_crop[:, 2]) / (im_tensor.size()[3] * 1.0)) + 1e-20  # y2-y1/h
            theta_crop[:, 1, 2] = ((sz_crop[:, 3] + sz_crop[:, 2]) / (im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h -1

        for i in range(im_tensor.size()[0]):
            try:
                grid_cropped = F.affine_grid(theta_crop[i].unsqueeze(0), (1, 1, int(sz_crop[i, 3]) - int(sz_crop[i, 2]) + 1, int(sz_crop[i, 1]) - int(sz_crop[i, 0]) + 1),
                                             align_corners=False)
                im_tensor_cropped = F.grid_sample(im_tensor[i].unsqueeze(0), grid_cropped, align_corners=False,
                                                  mode='bilinear')  # , padding_mode="border"
                img_crop[i, 0, int(sz_crop[i, 2]):int(sz_crop[i, 3]) + 1,
                int(sz_crop[i, 0]):int(sz_crop[i, 1]) + 1] = im_tensor_cropped
            except:
                print('negative values --- img_crop all zeros')

        return img_crop, theta_crop


class net3D(nn.Module):
    def __init__(self, ngpu,channel_dim, patch_size):
        super(net3D, self).__init__()
        self.ngpu = ngpu

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv3d(channel_dim, 8, kernel_size=7),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(8, 10, kernel_size=5),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * int((patch_size[0] / 4) - 4) * int((patch_size[1] / 4) - 4) * int((patch_size[2] / 4) - 4), 1024),
            nn.ReLU(True),
            nn.Linear(1024, 32),
            nn.ReLU(True),
            nn.Linear(32, 6),
            nn.ReLU(True)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([patch_size[0]/4, patch_size[0]*3/4, patch_size[1]/4, patch_size[1]*3/4,patch_size[2]/4, patch_size[2]*3/4], dtype=torch.float))


        # Spatial transformer network forward function
    def forward(self, im_tensor, mode = 'normal'):
        xs = self.localization(im_tensor)
        xs = xs.view(-1, 10 * int((im_tensor.size()[2] / 4) - 4) * int((im_tensor.size()[3] / 4) - 4) * int((im_tensor.size()[4] / 4) - 4))
        sz_crop = self.fc_loc(xs) #regressor for 3*3 matrix
        theta_crop = torch.zeros((im_tensor.size()[0],3,4),requires_grad=True, dtype=torch.float).to(torch.device("cuda"))
        '''
                PyTorch or more precisely autograd is not very good in handling in-place operations, 
                especially on those tensors with the requires_grad flag set to True.
                Generally you should avoid in-place operations where it is possible, in some cases it can work, 
                but you should always avoid in-place operations on tensors where you set requires_grad to True.
                Unfortunately there are not many pytorch functions to help out on this problem. 
                So you would have to use a helper tensor to avoid the in-place operation.
        '''
        theta_crop = theta_crop + 0 # new tensor, out-of-place operation (get around the problem)

        if mode == 'normal':
            for i in range(im_tensor.size()[0]):
                theta_crop[i,0,0] = ((sz_crop[i, 1] - sz_crop[i, 0])/(im_tensor.size()[2]*1.0)) + 1e-20 #y2-y1/w
                theta_crop[i,0,3] = ((sz_crop[i, 1] + sz_crop[i, 0])/(im_tensor.size()[2]*1.0)) - 1 #y2+y1/w - 1
                theta_crop[i,1,1] = ((sz_crop[i, 3] - sz_crop[i, 2])/(im_tensor.size()[3]*1.0))  + 1e-20 #x2-x1/h
                theta_crop[i,1,3] = ((sz_crop[i, 3] + sz_crop[i, 2])/(im_tensor.size()[3]*1.0)) - 1 #x2+x1/h -1
                theta_crop[i,2,2] = ((sz_crop[i, 5] - sz_crop[i, 4])/(im_tensor.size()[4]*1.0))  + 1e-20 #z2-z1/d
                theta_crop[i,2,3] = ((sz_crop[i, 5] + sz_crop[i, 4])/(im_tensor.size()[4]*1.0)) - 1 #z2+z1/d -1
        elif mode == 'security':
            for i in range(im_tensor.size()[0]):
                # adding safety margins of 0.2 of the size
                addx = int(im_tensor.size()[2] * 0.2)
                addy = int(im_tensor.size()[3] * 0.2)
                addz = int(im_tensor.size()[4] * 0.2)
                if (sz_crop[i, 1] - sz_crop[i, 0]) < (im_tensor.size()[2] - addx):
                    theta_crop[i, 0, 0] = ((sz_crop[i, 1] - sz_crop[i, 0] + addx) / (
                            im_tensor.size()[2] * 1.0)) + 1e-20  # x2-x1/w
                    if sz_crop[i, 0] < (addx / 2):
                        theta_crop[i, 0, 3] = ((sz_crop[i, 1] + addx - sz_crop[i, 0]) / (
                                im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    elif sz_crop[i, 1] > (im_tensor.size()[2] - (addx / 2)):
                        theta_crop[i, 0, 3] = ((im_tensor.size()[2] + sz_crop[i, 0] - addx + im_tensor.size()[2] -
                                                sz_crop[i, 1]) / (
                                                       im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    else:
                        theta_crop[i, 0, 3] = ((sz_crop[i, 1] + sz_crop[i, 0]) / (
                                im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                else:
                    theta_crop[i, 0, 0] = ((sz_crop[i, 1] - sz_crop[i, 0]) / (
                            im_tensor.size()[2] * 1.0)) + 1e-20  # x2-x1/w
                    theta_crop[i, 0, 3] = ((sz_crop[i, 1] + sz_crop[i, 0]) / (
                            im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1

                if (sz_crop[i, 3] - sz_crop[i, 2]) < (im_tensor.size()[3] - addy):
                    theta_crop[i, 1, 1] = ((sz_crop[i, 3] - sz_crop[i, 2] + addy) / (
                            im_tensor.size()[3] * 1.0)) + 1e-20  # y2-y1/h
                    if sz_crop[i, 2] < (addy / 2):
                        theta_crop[i, 1, 3] = ((sz_crop[i, 3] + addy - sz_crop[i, 2]) / (
                                im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h - 1
                    elif sz_crop[i, 3] > (im_tensor.size()[3] - (addy / 2)):
                        theta_crop[i, 1, 3] = ((im_tensor.size()[3] + sz_crop[i, 2] - addy + im_tensor.size()[3] -
                                                sz_crop[
                                                    i, 3]) / (im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h- 1
                    else:
                        theta_crop[i, 1, 3] = ((sz_crop[i, 3] + sz_crop[i, 2]) / (
                                im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h - 1
                else:
                    theta_crop[i, 1, 1] = ((sz_crop[i, 3] - sz_crop[i, 2]) / (
                            im_tensor.size()[3] * 1.0)) + 1e-20  # y2-y1/h
                    theta_crop[i, 1, 3] = ((sz_crop[i, 3] + sz_crop[i, 2]) / (
                            im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h - 1

                if (sz_crop[i, 5] - sz_crop[i, 4]) < (im_tensor.size()[4] - addz):
                    theta_crop[i, 2, 2] = ((sz_crop[i, 5] - sz_crop[i, 4] + addz) / (
                            im_tensor.size()[4] * 1.0)) + 1e-20  # z2-z1/d
                    if sz_crop[i, 4] < (addz / 2):
                        theta_crop[i, 2, 3] = ((sz_crop[i, 5] + addz - sz_crop[i, 4]) / (
                                im_tensor.size()[4] * 1.0)) - 1  # z2+z1/d - 1
                    elif sz_crop[i, 5] > (im_tensor.size()[4] - (addz / 2)):
                        theta_crop[i, 2, 3] = ((im_tensor.size()[4] + sz_crop[i, 4] - addz + im_tensor.size()[4] -
                                                sz_crop[
                                                    i, 5]) / (im_tensor.size()[4] * 1.0)) - 1  # z2+z1/d- 1
                    else:
                        theta_crop[i, 2, 3] = ((sz_crop[i, 5] + sz_crop[i, 4]) / (
                                im_tensor.size()[4] * 1.0)) - 1  # z2+z1/d - 1
                else:
                    theta_crop[i, 2, 2] = ((sz_crop[i, 5] - sz_crop[i, 4]) / (
                            im_tensor.size()[4] * 1.0)) + 1e-20  # z2-z1/d
                    theta_crop[i, 2, 3] = ((sz_crop[i, 5] + sz_crop[i, 4]) / (
                            im_tensor.size()[4] * 1.0)) - 1  # z2+z1/d - 1

        grid_cropped = F.affine_grid(theta_crop, im_tensor.size(), align_corners=False)
        im_tensor_cropped = F.grid_sample(im_tensor, grid_cropped, align_corners=False, mode='bilinear') #, padding_mode="border"

        return im_tensor_cropped, theta_crop
        
        
class net3D_new(nn.Module):
    def __init__(self, ngpu,channel_dim, patch_size):
        super(net3D_new, self).__init__()
        self.ngpu = ngpu

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv3d(channel_dim, 8, kernel_size=7),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(8, 10, kernel_size=5),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * int((patch_size[0] / 4) - 4) * int((patch_size[1] / 4) - 4) * int((patch_size[2] / 4) - 4), 1024),
            nn.ReLU(True),
            nn.Linear(1024, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([1, 1, 1, 0,0, 0], dtype=torch.float))


        # Spatial transformer network forward function
    def forward(self, im_tensor, mode = 'normal'):
        xs = self.localization(im_tensor)
        xs = xs.view(-1, 10 * int((im_tensor.size()[2] / 4) - 4) * int((im_tensor.size()[3] / 4) - 4) * int((im_tensor.size()[4] / 4) - 4))
        sz_crop = self.fc_loc(xs) #regressor for 3*3 matrix
        theta_crop = torch.zeros((im_tensor.size()[0],3,4),requires_grad=True, dtype=torch.float).to(torch.device("cuda"))
        '''
                PyTorch or more precisely autograd is not very good in handling in-place operations, 
                especially on those tensors with the requires_grad flag set to True.
                Generally you should avoid in-place operations where it is possible, in some cases it can work, 
                but you should always avoid in-place operations on tensors where you set requires_grad to True.
                Unfortunately there are not many pytorch functions to help out on this problem. 
                So you would have to use a helper tensor to avoid the in-place operation.
        '''
        theta_crop = theta_crop + 0 # new tensor, out-of-place operation (get around the problem)

        if mode == 'normal':
            for i in range(im_tensor.size()[0]):
                theta_crop[i,0,0] = F.sigmoid(sz_crop[i, 0])
                theta_crop[i,0,3] = F.tanh(sz_crop[i, 3])
                theta_crop[i,1,1] = F.sigmoid(sz_crop[i, 1])
                theta_crop[i,1,3] = F.tanh(sz_crop[i, 4])
                theta_crop[i,2,2] = F.sigmoid(sz_crop[i, 2])
                theta_crop[i,2,3] = F.tanh(sz_crop[i, 5])
        elif mode == 'security':
            for i in range(im_tensor.size()[0]):
                #adding safety margins of 0.2 of the size
                theta_crop[i,0,0] = F.sigmoid(sz_crop[i, 0])+0.1
                theta_crop[i,0,3] = F.tanh(sz_crop[i, 3])
                theta_crop[i,1,1] = F.sigmoid(sz_crop[i, 1])+0.1
                theta_crop[i,1,3] = F.tanh(sz_crop[i, 4])
                theta_crop[i,2,2] = F.sigmoid(sz_crop[i, 2])+0.1
                theta_crop[i,2,3] = F.tanh(sz_crop[i, 5])

        grid_cropped = F.affine_grid(theta_crop, im_tensor.size(), align_corners=False)
        im_tensor_cropped = F.grid_sample(im_tensor, grid_cropped, align_corners=False, mode='bilinear') #, padding_mode="border"

        return im_tensor_cropped, theta_crop


class net3D_2(nn.Module):
    def __init__(self, ngpu, channel_dim, patch_size, use_bias = True):
        super(net3D_2, self).__init__()
        self.ngpu = ngpu

        # Spatial transformer localization-network
        self.loc1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(8),
            nn.ReLU(True)
        )
        self.loc2 = nn.Sequential(
            nn.Conv3d(9, 8, 3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm3d(8),
            nn.ReLU(True)
        )

        self.loc3 = nn.Sequential(
            nn.Conv3d(8, 16, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(16),
            nn.ReLU(True)
        )
        self.loc4 = nn.Sequential(
            nn.Conv3d(24, 16, 3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm3d(16),
            nn.ReLU(True)
        )
        self.loc5 = nn.Sequential(
            nn.Conv3d(16, 24, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(24),
            nn.ReLU(True)
        )
        self.loc6 = nn.Sequential(
            nn.Conv3d(40, 24, 3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm3d(24),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(24 * int(patch_size[0] / 8) * int(patch_size[1] / 8)* int(patch_size[2] / 8), 1024),
            nn.ReLU(True),
            nn.Linear(1024, 6),
            nn.Hardtanh(min_val=0.0, max_val=(patch_size[0] - 1), inplace=True)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([patch_size[0]/4, patch_size[0]*3/4, patch_size[1]/4, patch_size[1]*3/4,patch_size[2]/4, patch_size[2]*3/4], dtype=torch.float))

        # Spatial transformer network forward function

    def forward(self, im_tensor, mode='normal'):
        x1 = self.loc1(im_tensor)
        x2 = self.loc2(torch.cat((x1, im_tensor),1))
        x3 = self.loc3(x2)
        x4 = self.loc4(torch.cat((x3, x2),1))
        x5 = self.loc5(x4)
        xs = self.loc6(torch.cat((x5, x4),1))

        xs = xs.view(-1, 24 * int(im_tensor.size()[2] / 8)* int(im_tensor.size()[3] / 8)* int(im_tensor.size()[4] / 8))
        sz_crop = self.fc_loc(xs)  # regressor for 2*2 matrix
        theta_crop = torch.zeros((im_tensor.size()[0], 3, 4), requires_grad=True, dtype=torch.float).to(
            torch.device("cuda"))
        theta_crop = theta_crop + 0
        img_crop = torch.zeros((im_tensor.size()), requires_grad=True, dtype=torch.float).to(torch.device("cuda"))
        img_crop = img_crop + im_tensor.min()

        if mode == 'normal':
                theta_crop[:,0,0] = ((sz_crop[:, 1] - sz_crop[:, 0])/(im_tensor.size()[2]*1.0)) + 1e-20 #x2-x1/w
                theta_crop[:,0,3] = ((sz_crop[:, 1] + sz_crop[:, 0])/(im_tensor.size()[2]*1.0)) - 1 #x2+x1/w - 1
                theta_crop[:,1,1] = ((sz_crop[:, 3] - sz_crop[:, 2])/(im_tensor.size()[3]*1.0))  + 1e-20 #y2-y1/h
                theta_crop[:,1,3] = ((sz_crop[:, 3] + sz_crop[:, 2])/(im_tensor.size()[3]*1.0)) - 1 #y2+y1/h -1
                theta_crop[:,2,2] = ((sz_crop[:, 5] - sz_crop[:, 4])/(im_tensor.size()[4]*1.0))  + 1e-20 #z2-z1/d
                theta_crop[:,2,3] = ((sz_crop[:, 5] + sz_crop[:, 4])/(im_tensor.size()[4]*1.0)) - 1 #z2+z1/d -1
        elif mode == 'security':
            for i in range(im_tensor.size()[0]):
                # adding safety margins of 0.2 of the size
                addz = int(im_tensor.size()[2] * 0.2)
                addx = int(im_tensor.size()[3] * 0.2)
                addy = int(im_tensor.size()[4] * 0.2)
                if (sz_crop[i, 1] - sz_crop[i, 0]) < (im_tensor.size()[2] - addz):
                    if sz_crop[i, 0] < (addz / 2):
                        sz_crop[i, 1] += (addz / 2)
                    elif sz_crop[i, 1] > (im_tensor.size()[2] - (addz / 2)):
                        sz_crop[i, 0] -= (addz / 2)
                    else:
                        sz_crop[i, 1] += (addz / 2)
                        sz_crop[i, 0] -= (addz / 2)
                if (sz_crop[i, 3] - sz_crop[i, 2]) < (im_tensor.size()[3] - addx):
                    if sz_crop[i, 2] < (addx / 2):
                        sz_crop[i, 3] += (addx / 2)
                    elif sz_crop[i, 3] > (im_tensor.size()[3] - (addx / 2)):
                        sz_crop[i, 2] -= (addx / 2)
                    else:
                        sz_crop[i, 3] += (addx / 2)
                        sz_crop[i, 2] -= (addx / 2)
                if (sz_crop[i, 5] - sz_crop[i, 4]) < (im_tensor.size()[4] - addy):
                    if sz_crop[i, 4] < (addy / 2):
                        sz_crop[i, 5] += (addy / 2)
                    elif sz_crop[i, 5] > (im_tensor.size()[4] - (addy / 2)):
                        sz_crop[i, 4] -= (addy / 2)
                    else:
                        sz_crop[i, 5] += (addy / 2)
                        sz_crop[i, 4] -= (addy / 2)

            theta_crop[:, 0, 0] = ((sz_crop[:, 1] - sz_crop[:, 0]) / (im_tensor.size()[2] * 1.0)) + 1e-20  # x2-x1/w
            theta_crop[:, 0, 3] = ((sz_crop[:, 1] + sz_crop[:, 0]) / (im_tensor.size()[2] * 1.0)) - 1  # x2+x1/w - 1
            theta_crop[:, 1, 1] = ((sz_crop[:, 3] - sz_crop[:, 2]) / (im_tensor.size()[3] * 1.0)) + 1e-20  # y2-y1/h
            theta_crop[:, 1, 3] = ((sz_crop[:, 3] + sz_crop[:, 2]) / (im_tensor.size()[3] * 1.0)) - 1  # y2+y1/h -1
            theta_crop[:, 2, 2] = ((sz_crop[:, 5] - sz_crop[:, 4]) / (im_tensor.size()[4] * 1.0)) + 1e-20  # z2-z1/d
            theta_crop[:, 2, 3] = ((sz_crop[:, 5] + sz_crop[:, 4]) / (im_tensor.size()[4] * 1.0)) - 1  # z2+z1/d -1

        for i in range(im_tensor.size()[0]):
            try:
                grid_cropped = F.affine_grid(theta_crop[i].unsqueeze(0), (1, 1, int(sz_crop[i, 1]) - int(sz_crop[i, 0]) + 1, int(sz_crop[i, 5]) - int(sz_crop[i, 4]) + 1, int(sz_crop[i, 3]) - int(sz_crop[i, 2]) + 1),
                                             align_corners=False)
                im_tensor_cropped = F.grid_sample(im_tensor[i].unsqueeze(0), grid_cropped, align_corners=False,
                                                  mode='bilinear')  # , padding_mode="border"

                img_crop[i:i+1, :, int(sz_crop[i, 0]):int(sz_crop[i, 1])+1,int(sz_crop[i, 4]): int(sz_crop[i, 5])+1, int(sz_crop[i, 2]):int(sz_crop[i, 3])+1] = im_tensor_cropped
            except:
                print('negative values --- img_crop all zeros')

        return img_crop, theta_crop, sz_crop
