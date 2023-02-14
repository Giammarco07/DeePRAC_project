import os
import numpy as np
import torch.nn as nn

class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out
        
class RED_CNN_3D(nn.Module):
    def __init__(self, channel_dim,out_ch=96):
        super(RED_CNN_3D, self).__init__()
        self.conv1 = nn.Conv3d(1, out_ch, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv4 = nn.Conv3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv5 = nn.Conv3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True)

        self.tconv1 = nn.ConvTranspose3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True)
        self.tconv2 = nn.ConvTranspose3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True)
        self.tconv3 = nn.ConvTranspose3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True)
        self.tconv4 = nn.ConvTranspose3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True)
        self.tconv5 = nn.ConvTranspose3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_ch)
        self.lrelu = nn.LeakyReLU()

        self.outputblock = nn.Sequential(
            nn.Conv3d(out_ch, channel_dim, 1, 1, 0, bias=True),
            nn.Softmax(dim=1),
        )

        self.outputblock1 = nn.Sequential(
            nn.ConvTranspose3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True),
            nn.Conv3d(out_ch, channel_dim, 1, 1, 0, bias=True),
            nn.Softmax(dim=1),
        )

        self.outputblock2 = nn.Sequential(
            nn.ConvTranspose3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ConvTranspose3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True),
            nn.Conv3d(out_ch, channel_dim, 1, 1, 0, bias=True),
            nn.Softmax(dim=1),
        )

        self.outputblock3 = nn.Sequential(
            nn.ConvTranspose3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ConvTranspose3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ConvTranspose3d(out_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=True),
            nn.Conv3d(out_ch, channel_dim, 1, 1, 0, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # encoder
        out = self.relu(self.bn(self.conv1(x)))
        residual_1 = out
        out = self.relu(self.bn(self.conv2(out)))
        residual_2 = out
        out = self.relu(self.bn(self.conv3(out)))
        residual_3 = out
        out = self.relu(self.bn(self.conv4(out)))
        residual_4 = out
        out = self.relu(self.bn(self.conv5(out)))
        # decoder
        out = self.bn(self.tconv1(out))
        out = out + residual_4
        out3 = self.bn(self.tconv2(self.lrelu(out)))
        out3 = out3 + residual_3
        out2 = self.bn(self.tconv3(self.lrelu(out3)))
        out2 = out2 + residual_2
        out1 = self.bn(self.tconv4(self.lrelu(out2)))
        out1 = out1 + residual_1
        out = self.bn(self.tconv5(self.lrelu(out1)))

        out0 = self.outputblock(self.lrelu(out))
        out1 = self.outputblock1(self.lrelu(out1))
        out2 = self.outputblock2(self.lrelu(out2))
        out3 = self.outputblock3(self.lrelu(out3))
        return out0, out1, out2, out3
