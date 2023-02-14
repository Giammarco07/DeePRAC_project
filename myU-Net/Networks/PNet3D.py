import torch.nn as nn
import torch



class net_64(nn.Module):
    def __init__(self, ndf=32, use_bias=False, in_c=1):
        super(net_64, self).__init__()

        self.downsample = nn.MaxPool3d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.convblock1 = nn.Sequential(
            nn.Conv3d(in_c, ndf, 3, 1, 1, bias=use_bias, dilation=1),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, ndf, (3, 3, 1), 1, (1, 1, 0), bias=use_bias, dilation=1),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv3d(ndf, ndf, 3, 1, 2, bias=use_bias, dilation=2),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, ndf, (3, 3, 1), 1, (2, 2, 0), bias=use_bias, dilation=2),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv3d(ndf, ndf, 3, 1, 3, bias=use_bias, dilation=3),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, ndf, (3, 3, 1), 1, (3, 3, 0), bias=use_bias, dilation=3),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, ndf, (3, 3, 1), 1, (3, 3, 0), bias=use_bias, dilation=3),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True)
        )
        self.convblock4 = nn.Sequential(
            nn.Conv3d(ndf, ndf, 3, 1, 4, bias=use_bias, dilation=4),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, ndf, (3, 3, 1), 1, (4, 4, 0), bias=use_bias, dilation=4),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, ndf, (3, 3, 1), 1, (4, 4, 0), bias=use_bias, dilation=4),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True)
        )
        self.convblock5 = nn.Sequential(
            nn.Conv3d(ndf, ndf, 3, 1, 5, bias=use_bias, dilation=5),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, ndf, (3, 3, 1), 1, (5, 5, 0), bias=use_bias, dilation=5),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, ndf, (3, 3, 1), 1, (5, 5, 0), bias=use_bias, dilation=5),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv3d((ndf // 4) * 5, ndf, 1, bias=use_bias),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, 1, 1, bias=use_bias),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )

        self.concatenate = nn.Sequential(
            nn.Conv3d(ndf, ndf // 4, 1, bias=use_bias),
            nn.BatchNorm3d(ndf // 4),
            nn.ReLU(inplace=True)
        )

        self.outputblock = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
            # nn.ReLU(True)
            # nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        encoder1 = self.convblock1(input)
        encoder2 = self.convblock2(encoder1)
        encoder3 = self.convblock3(encoder2)
        encoder4 = self.convblock4(encoder3)
        encoder5 = self.convblock5(encoder4)
        concatenation = torch.cat([self.concatenate(encoder1), self.concatenate(encoder2), self.concatenate(encoder3),
                                   self.concatenate(encoder4), self.concatenate(encoder5)], 1)

        encoder6 = self.convblock6(concatenation)
        output = self.outputblock(encoder6)

        return output