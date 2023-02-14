import torch.nn as nn
import torch
import torchvision.models as models

class net(nn.Module):
    def __init__(self, ngpu, channel_dim, use_bias = False):
        super(net, self).__init__()
        print('use bias in the network: ',use_bias)
        self.ngpu = ngpu
        self.net = models.vgg16(pretrained=True).features

        self.specialized_block1 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1, bias=use_bias),
            nn.ReLU(True)
        )
        self.specialized_block2 = nn.Sequential(
            nn.Conv2d(128, 16, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        )
        self.specialized_block3 = nn.Sequential(
            nn.Conv2d(256, 16, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        )
        self.specialized_block4 = nn.Sequential(
            nn.Conv2d(512, 16, 3, 1, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        )
        self.outputblock = nn.Sequential(
            nn.Conv2d(16 * 5, channel_dim, 1, 1, 0, bias=False),
            nn.Softmax(dim=1)
        )

        torch.manual_seed(786)
        nn.init.kaiming_normal_(self.specialized_block1[0].weight.data, a=0, mode='fan_in', nonlinearity='relu')
        self.specialized_block1[0].bias.data.zero_()
        nn.init.kaiming_normal_(self.specialized_block2[0].weight.data, a=0, mode='fan_in', nonlinearity='relu')
        self.specialized_block2[0].bias.data.zero_()
        nn.init.kaiming_normal_(self.specialized_block3[0].weight.data, a=0, mode='fan_in', nonlinearity='relu')
        self.specialized_block3[0].bias.data.zero_()
        nn.init.kaiming_normal_(self.specialized_block4[0].weight.data, a=0, mode='fan_in', nonlinearity='relu')
        self.specialized_block4[0].bias.data.zero_()
        nn.init.kaiming_normal_(self.outputblock[0].weight.data, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, input):
        first = self.net[0:4](input)
        second = self.net[4:9](first)
        third = self.net[9:16](second)
        forth = self.net[16:23](third)
        fifth = self.net[23:30](forth)
        up1 = self.specialized_block1(first)
        up2 = self.specialized_block2(second)
        up3 = self.specialized_block3(third)
        up4 = self.specialized_block4(forth)
        up5 = self.specialized_block4(fifth)
        output = torch.cat([up1, up2, up3, up4, up5], 1)
        output = self.outputblock(output)

        return output


