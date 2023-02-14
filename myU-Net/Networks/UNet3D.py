import torch.nn as nn
import torch


class net(nn.Module):
    def __init__(self, ngpu, channel_dim, ndf=30, use_bias=False):
        super(net, self).__init__()
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            # state size: 1 * 128 * 128 * 128
            nn.Conv3d(1, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace=True),
            # state size: ndf * 128 * 128 * 128
        )
        self.convblock2 = nn.Sequential(
            # state size: ndf * 128 * 128 * 128
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace=True),
            # state size: ndf * 128 * 128 * 128
        )
        self.convblock3 = nn.Sequential(
            # state size: ndf * 128 * 128 * 128
            nn.Conv3d(ndf, ndf * 2, 3, stride=(2, 2, 2), padding=1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*2 * 64 * 64 * 64
        )
        self.convblock4 = nn.Sequential(
            # state size: ndf*2 * 64 * 64 * 64
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*2 * 64 * 64 * 64
        )
        self.convblock5 = nn.Sequential(
            # state size: ndf*2 * 64 * 64 * 64
            nn.Conv3d(ndf * 2, ndf * 4, 3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*4 * 32 * 32 * 32
        )
        self.convblock6 = nn.Sequential(
            # state size: ndf*4 * 32 * 32 * 32
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*4 * 32 * 32 * 32
        )
        self.convblock7 = nn.Sequential(
            # state size: ndf*4 * 32 * 32 * 32
            nn.Conv3d(ndf * 4, ndf * 8, 3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*8 * 16 * 16 * 16
        )
        self.convblock8 = nn.Sequential(
            # state size: ndf*8 * 16 * 16 * 16
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*8 * 16 * 16 * 16
        )
        self.convblock9 = nn.Sequential(
            # state size: ndf*8 * 16 * 16 * 16
            nn.Conv3d(ndf * 8, 320, 3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace=True),
            # state size: 320 * 8 * 8 * 8
        )
        self.convblock10 = nn.Sequential(
            # state size: 320 * 8 * 8 * 8
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace=True),
            # state size: 320 * 8 * 8 * 8
        )

        self.bridgeblock1 = nn.Sequential(
            # state size: 320 * 8 * 8 * 8
            nn.Conv3d(320, 320, 3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace=True),
            # state size: 320 * 4 * 4 * 4
        )
        self.bridgeblock2 = nn.Sequential(
            # state size: 320 * 4 * 4 * 4
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(320, 320, kernel_size=2, stride=2, bias=use_bias),
            # state size: 320 * 8 * 8 * 8
        )

        self.deconvblock1 = nn.Sequential(
            # state size: (cat : 320+320) * 8 * 8 * 8
            nn.Conv3d(640, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace=True),
            # state size: 320 * 8 * 8 * 8
        )
        self.deconvblock2 = nn.Sequential(
            # state size: 320 * 8 * 8 * 8
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(320, ndf * 8, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*8 * 16 * 16 * 16
        )
        self.deconvblock3 = nn.Sequential(
            # state size: (cat : ndf*8+ndf*8) * 16 * 16 * 16
            nn.Conv3d(ndf * 8 * 2, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*8 * 16 * 16 * 16
        )
        self.deconvblock4 = nn.Sequential(
            # state size: ndf*8 * 16 * 16 * 16
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace=True), )
        self.deconvblock4_T = nn.ConvTranspose3d(ndf * 8, ndf * 4, kernel_size=2, stride=2, bias=use_bias)
        # state size: ndf*4 * 32 * 32 * 32

        self.deconvblock5 = nn.Sequential(
            # state size: (cat : ndf*4+ndf*4) * 32 * 32 * 32
            nn.Conv3d(ndf * 4 * 2, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*4 * 32 * 32 * 32
        )
        self.deconvblock6 = nn.Sequential(
            # state size: ndf*4 * 32 * 32 * 32
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace=True), )
        self.deconvblock6_T = nn.ConvTranspose3d(ndf * 4, ndf * 2, kernel_size=2, stride=2, bias=use_bias)
        # state size: ndf*2 * 64 * 64 * 64

        self.deconvblock7 = nn.Sequential(
            # state size: (cat : ndf*2+ndf*2) *  64 * 64 * 64
            nn.Conv3d(ndf * 2 * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*2 *  64 * 64 * 64
        )
        self.deconvblock8 = nn.Sequential(
            # state size: ndf*2 *  64 * 64 * 64
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace=True), )
        self.deconvblock8_T = nn.ConvTranspose3d(ndf * 2, ndf, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=use_bias)
        # state size: ndf *  128 * 128 * 128

        self.deconvblock9 = nn.Sequential(
            # state size: (cat : ndf+ndf) *  128 * 128 * 128
            nn.Conv3d(ndf * 2, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace=True),
            # state size: ndf *  128 * 128 * 128
        )
        self.deconvblock10 = nn.Sequential(
            # state size: ndf *  128 * 128 * 128
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace=True),
            # state size: ndf *  128 * 128 * 128
        )

        self.outputblock = nn.Sequential(
            # state size: ndf * 128 * 128 * 128
            nn.Conv3d(ndf, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid()
            # state size: channel_dim * 128 * 128 * 128
        )

        self.outputblock1 = nn.Sequential(
            # state size: ndf*2 * 64*64*64
            nn.Conv3d(ndf * 2, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            # state size: channel_dim * 128 * 128 * 128
        )

        self.outputblock2 = nn.Sequential(
            # state size: ndf*4 * 32*32*32
            nn.Conv3d(ndf * 4, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
            # state size: channel_dim * 128 * 128 * 128
        )

        self.outputblock3 = nn.Sequential(
            # state size: ndf*8 * 16*16*16
            nn.Conv3d(ndf * 8, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False)
            # state size: channel_dim * 128 * 128 * 128
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        # for now it only supports one GPU
        encoder1 = self.convblock1(input)
        encoder2 = self.convblock2(encoder1)
        encoder3 = self.convblock3(encoder2)
        encoder4 = self.convblock4(encoder3)
        encoder5 = self.convblock5(encoder4)
        encoder6 = self.convblock6(encoder5)
        encoder7 = self.convblock7(encoder6)
        encoder8 = self.convblock8(encoder7)
        encoder9 = self.convblock9(encoder8)
        encoder10 = self.convblock10(encoder9)

        bridge1 = self.bridgeblock1(encoder10)
        bridge2 = self.bridgeblock2(bridge1)

        skip1 = torch.cat([bridge2, encoder10], 1)
        decoder1 = self.deconvblock1(skip1)
        decoder2 = self.deconvblock2(decoder1)
        skip2 = torch.cat([decoder2, encoder8], 1)
        decoder3 = self.deconvblock3(skip2)
        decoder4_out = self.deconvblock4(decoder3)
        output3 = self.outputblock3(decoder4_out)
        decoder4 = self.deconvblock4_T(decoder4_out)
        skip3 = torch.cat([decoder4, encoder6], 1)
        decoder5 = self.deconvblock5(skip3)
        decoder6_out = self.deconvblock6(decoder5)
        output2 = self.outputblock2(decoder6_out)
        decoder6 = self.deconvblock6_T(decoder6_out)
        skip4 = torch.cat([decoder6, encoder4], 1)
        decoder7 = self.deconvblock7(skip4)
        decoder8_out = self.deconvblock8(decoder7)
        output1 = self.outputblock1(decoder8_out)
        decoder8 = self.deconvblock8_T(decoder8_out)
        skip5 = torch.cat([decoder8, encoder2], 1)
        decoder9 = self.deconvblock9(skip5)
        decoder10 = self.deconvblock10(decoder9)

        output = self.outputblock(decoder10)

        return output, output1, output2, output3


class net_64(nn.Module):
    def __init__(self, ngpu, channel_dim, ndf=30, use_bias=False, in_c=1, mode='seg'):
        super(net_64, self).__init__()
        self.ngpu = ngpu
        self.mode = mode
        self.convblock1 = nn.Sequential(
            # state size: in_c * 64
            nn.Conv3d(in_c, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace=True),
            # state size: ndf * 64
        )
        self.convblock2 = nn.Sequential(
            # state size: ndf * 64
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace=True),
            # state size: ndf * 64
        )
        self.convblock3 = nn.Sequential(
            # state size: ndf * 64
            nn.Conv3d(ndf, ndf * 2, 3, stride=(2, 2, 2), padding=1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*2 * 32
        )
        self.convblock4 = nn.Sequential(
            # state size: ndf*2 * 32
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*2 * 32
        )
        self.convblock5 = nn.Sequential(
            # state size: ndf*2 * 32
            nn.Conv3d(ndf * 2, ndf * 4, 3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*4 * 16
        )
        self.convblock6 = nn.Sequential(
            # state size: ndf*4 * 16
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*4 * 16
        )
        self.convblock7 = nn.Sequential(
            # state size: ndf*4 * 16
            nn.Conv3d(ndf * 4, ndf * 8, 3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*8 * 8
        )
        self.convblock8 = nn.Sequential(
            # state size: ndf*8 * 8
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*8 * 8
        )

        self.bridgeblock1 = nn.Sequential(
            # state size: ndf * 8 * 8 * 8 * 8
            nn.Conv3d(ndf * 8, 320, 3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace=True),
            # state size: 320 * 4 * 4 * 4
        )
        self.bridgeblock2 = nn.Sequential(
            # state size: 320 * 4 * 4 * 4
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(320, ndf * 8, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf * 8 * 8 * 8 * 8
        )

        self.deconvblock3 = nn.Sequential(
            # state size: (cat : ndf*8+ndf*8) * 8
            nn.Conv3d(ndf * 8 * 2, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*8 * 8
        )
        self.deconvblock4 = nn.Sequential(
            # state size: ndf*8 * 8
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace=True), )
        self.deconvblock4_T = nn.ConvTranspose3d(ndf * 8, ndf * 4, kernel_size=2, stride=2, bias=use_bias)
        # state size: ndf*4 * 16

        self.deconvblock5 = nn.Sequential(
            # state size: (cat : ndf*4+ndf*4) * 16
            nn.Conv3d(ndf * 4 * 2, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*4 * 16
        )
        self.deconvblock6 = nn.Sequential(
            # state size: ndf*4 * 16
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace=True), )
        self.deconvblock6_T = nn.ConvTranspose3d(ndf * 4, ndf * 2, kernel_size=2, stride=2, bias=use_bias)
        # state size: ndf*2 * 32

        self.deconvblock7 = nn.Sequential(
            # state size: (cat : ndf*2+ndf*2) *  32
            nn.Conv3d(ndf * 2 * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*2 *  32
        )
        self.deconvblock8 = nn.Sequential(
            # state size: ndf*2 *  32
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace=True), )
        self.deconvblock8_T = nn.ConvTranspose3d(ndf * 2, ndf, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=use_bias)
        # state size: ndf *  64

        self.deconvblock9 = nn.Sequential(
            # state size: (cat : ndf+ndf) *  64
            nn.Conv3d(ndf * 2, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace=True),
            # state size: ndf *  64
        )
        self.deconvblock10 = nn.Sequential(
            # state size: ndf *  64
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace=True),
            # state size: ndf *  64
        )

        self.outputblock = nn.Sequential(
            # state size: ndf * 64
            nn.Conv3d(ndf, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid()
            # state size: channel_dim * 64
        )

        self.outputblock1 = nn.Sequential(
            # state size: ndf*2 * 32
            nn.Conv3d(ndf * 2, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            # state size: channel_dim * 64
        )

        self.outputblock2 = nn.Sequential(
            # state size: ndf*4 * 16
            nn.Conv3d(ndf * 4, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
            # state size: channel_dim * 64
        )

        self.outputblock3 = nn.Sequential(
            # state size: ndf*8 * 8
            nn.Conv3d(ndf * 8, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False)
            # state size: channel_dim * 64
        )

        if self.mode == 'class':
            self.classblock = nn.Sequential(
                nn.Linear(ndf * 32 * 64 * 64, 128),
                nn.ReLU(True),
                nn.Linear(128, 1),
                # nn.ReLU(True)
                # nn.Sigmoid()
            )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        # for now it only supports one GPU
        encoder1 = self.convblock1(input)
        encoder2 = self.convblock2(encoder1)
        encoder3 = self.convblock3(encoder2)
        encoder4 = self.convblock4(encoder3)
        encoder5 = self.convblock5(encoder4)
        encoder6 = self.convblock6(encoder5)
        encoder7 = self.convblock7(encoder6)
        encoder8 = self.convblock8(encoder7)

        bridge1 = self.bridgeblock1(encoder8)
        bridge2 = self.bridgeblock2(bridge1)

        skip1 = torch.cat([bridge2, encoder8], 1)
        decoder3 = self.deconvblock3(skip1)
        decoder4_out = self.deconvblock4(decoder3)
        output3 = self.outputblock3(decoder4_out)
        decoder4 = self.deconvblock4_T(decoder4_out)
        skip3 = torch.cat([decoder4, encoder6], 1)
        decoder5 = self.deconvblock5(skip3)
        decoder6_out = self.deconvblock6(decoder5)
        output2 = self.outputblock2(decoder6_out)
        decoder6 = self.deconvblock6_T(decoder6_out)
        skip4 = torch.cat([decoder6, encoder4], 1)
        decoder7 = self.deconvblock7(skip4)
        decoder8_out = self.deconvblock8(decoder7)
        output1 = self.outputblock1(decoder8_out)
        decoder8 = self.deconvblock8_T(decoder8_out)
        skip5 = torch.cat([decoder8, encoder2], 1)
        decoder9 = self.deconvblock9(skip5)
        decoder10 = self.deconvblock10(decoder9)

        if self.mode == 'seg':

            output = self.outputblock(decoder10)

            return output, output1, output2, output3

        elif self.mode == 'class':

            output = self.classblock(torch.flatten(decoder10, 1))

            return output

        else:
            print('mode: ', self.mode, 'is not defined.')

class net_new(nn.Module):
    def __init__(self, ngpu, channel_dim, ndf = 30, use_bias = False, in_c=1):
        super(net_new, self).__init__()
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            #state size: 1 * 96 * 160 * 160
            nn.Conv3d(in_c, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 96 * 160 * 160
        )
        self.convblock2 = nn.Sequential(
            # state size: ndf * 96 * 160 * 160
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 96 * 160 * 160
        )
        self.convblock3 = nn.Sequential(
            # state size: ndf * 96 * 160 * 160
            nn.Conv3d(ndf, ndf * 2, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 48 * 80 * 80
        )
        self.convblock4 = nn.Sequential(
            # state size: ndf*2 * 48 * 80 * 80
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 48 * 80 * 80
        )
        self.convblock5 = nn.Sequential(
            # state size: ndf*2 * 48 * 80 * 80
            nn.Conv3d(ndf*2, ndf * 4, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*4 * 24 * 40 * 40
        )
        self.convblock6 = nn.Sequential(
            # state size: ndf*4 * 24 * 40 * 40
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*4 * 24 * 40 * 40
        )
        self.convblock7 = nn.Sequential(
            # state size: ndf*4 * 24 * 40 * 40
            nn.Conv3d(ndf*4, ndf * 8, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*8 * 12 * 20 * 20
        )
        self.convblock8 = nn.Sequential(
            # state size: ndf*8 *  12 * 20 * 20
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*8 *  12 * 20 * 20
        )
        self.convblock9 = nn.Sequential(
            # state size: ndf*8 *  12 * 20 * 20
            nn.Conv3d(ndf*8, 320, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 6 * 10 * 10
        )
        self.convblock10 = nn.Sequential(
            # state size: 320 * 6 * 10 * 10
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 6 * 10 * 10
        )

        self.bridgeblock1= nn.Sequential(
            # state size: 320 * 6 * 10 * 10
            nn.Conv3d(320, 320, 3, stride = (1,2,2), padding=1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 6 * 5 * 5
        )
        self.bridgeblock2 = nn.Sequential(
            # state size: 320 * 6 * 5 * 5
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            nn.ConvTranspose3d(320, 320, kernel_size=(1,2,2), stride = (1,2,2), bias=use_bias),
            # state size: 320 * 6 * 10 * 10
        )

        self.deconvblock1 = nn.Sequential(
            # state size: (cat : 320+320) *  12 * 10 * 10
            nn.Conv3d(640, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 *  12 * 10 * 10
        )
        self.deconvblock2 = nn.Sequential(
            # state size: 320 *  12 * 10 * 10
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            nn.ConvTranspose3d(320, ndf*8, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*8 * 24 * 20 * 20
        )
        self.deconvblock3 = nn.Sequential(
            # state size: (cat : ndf*8+ndf*8) * 24 * 20 * 20
            nn.Conv3d(ndf*8*2, ndf*8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*8),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*8 * 24 * 20 * 20
        )
        self.deconvblock4 = nn.Sequential(
            # state size: ndf*8 * 24 * 20 * 20
            nn.Conv3d(ndf*8, ndf*8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*8),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock4_T = nn.ConvTranspose3d(ndf*8, ndf*4, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*4 * 48 * 40 * 40

        self.deconvblock5 = nn.Sequential(
            # state size: (cat : ndf*4+ndf*4) * 48 * 40 * 40
            nn.Conv3d(ndf*4 * 2, ndf*4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*4),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*4 * 48 * 40 * 40
        )
        self.deconvblock6 = nn.Sequential(
            # state size: ndf*4 * 48 * 40 * 40
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock6_T = nn.ConvTranspose3d(ndf * 4, ndf * 2, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*2 * 96 * 80 * 80

        self.deconvblock7 = nn.Sequential(
            # state size: (cat : ndf*2+ndf*2) *    96 * 80 * 80
            nn.Conv3d(ndf * 2 * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 *    96 * 80 * 80
        )
        self.deconvblock8 = nn.Sequential(
            # state size: ndf*2 *    96 * 80 * 80
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock8_T = nn.ConvTranspose3d(ndf * 2, ndf, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf *  96 * 160 * 160

        self.deconvblock9 = nn.Sequential(
            # state size: (cat : ndf+ndf) *   96 * 160 * 160
            nn.Conv3d(ndf * 2, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf *   96 * 160 * 160
        )
        self.deconvblock10 = nn.Sequential(
            # state size: ndf *   96 * 160 * 160
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf *   96 * 160 * 160
        )

        self.outputblock = nn.Sequential(
            # state size: ndf *  96 * 160 * 160
            nn.Conv3d(ndf, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid()
            # state size: channel_dim * 96 * 160 * 160
        )

        self.outputblock1 = nn.Sequential(
            # state size: ndf*2 * 48 * 80 * 80use_bias
            nn.Conv3d(ndf * 2, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False)
            # state size: channel_dim * 96 * 160 * 160
        )

        self.outputblock2 = nn.Sequential(
            # state size: ndf*4 * 24*40*40
            nn.Conv3d(ndf * 4, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=(4,4,4), mode='trilinear', align_corners=False)
            # state size: channel_dim * 96 * 160 * 160
        )

        self.outputblock3 = nn.Sequential(
            # state size: ndf*8 * 12 * 20* 20
            nn.Conv3d(ndf * 8, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=(8,8,8), mode='trilinear', align_corners=False)
            # state size: channel_dim * 96 * 160 * 160
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        # for now it only supports one GPU
            encoder1 = self.convblock1(input)
            encoder2 = self.convblock2(encoder1)
            encoder3 = self.convblock3(encoder2)
            encoder4 = self.convblock4(encoder3)
            encoder5 = self.convblock5(encoder4)
            encoder6 = self.convblock6(encoder5)
            encoder7 = self.convblock7(encoder6)
            encoder8 = self.convblock8(encoder7)
            encoder9 = self.convblock9(encoder8)
            encoder10 = self.convblock10(encoder9)

            bridge1 = self.bridgeblock1(encoder10)
            bridge2 = self.bridgeblock2(bridge1)

            skip1 = torch.cat([bridge2, encoder10], 1)
            decoder1 = self.deconvblock1(skip1)
            decoder2 = self.deconvblock2(decoder1)
            skip2 = torch.cat([decoder2, encoder8], 1)
            decoder3 = self.deconvblock3(skip2)
            decoder4_out = self.deconvblock4(decoder3)
            output3 = self.outputblock3(decoder4_out)
            decoder4 = self.deconvblock4_T(decoder4_out)
            skip3 = torch.cat([decoder4, encoder6], 1)
            decoder5 = self.deconvblock5(skip3)
            decoder6_out = self.deconvblock6(decoder5)
            output2 = self.outputblock2(decoder6_out)
            decoder6 = self.deconvblock6_T(decoder6_out)
            skip4 = torch.cat([decoder6, encoder4], 1)
            decoder7 = self.deconvblock7(skip4)
            decoder8_out = self.deconvblock8(decoder7)
            output1 = self.outputblock1(decoder8_out)
            decoder8 = self.deconvblock8_T(decoder8_out)
            skip5 = torch.cat([decoder8, encoder2], 1)
            decoder9 = self.deconvblock9(skip5)
            decoder10 = self.deconvblock10(decoder9)

            output = self.outputblock(decoder10)

            return output, output1, output2, output3

class net_ddt(nn.Module):
    def __init__(self, ngpu, channel_dim, k, ndf = 30, use_bias = False):
        super(net_ddt, self).__init__()
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            #state size: 1 * 96 * 160 * 160
            nn.Conv3d(1, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 96 * 160 * 160
        )
        self.convblock2 = nn.Sequential(
            # state size: ndf * 96 * 160 * 160
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 96 * 160 * 160
        )
        self.convblock3 = nn.Sequential(
            # state size: ndf * 96 * 160 * 160
            nn.Conv3d(ndf, ndf * 2, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 48 * 80 * 80
        )
        self.convblock4 = nn.Sequential(
            # state size: ndf*2 * 48 * 80 * 80
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 48 * 80 * 80
        )
        self.convblock5 = nn.Sequential(
            # state size: ndf*2 * 48 * 80 * 80
            nn.Conv3d(ndf*2, ndf * 4, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*4 * 24 * 40 * 40
        )
        self.convblock6 = nn.Sequential(
            # state size: ndf*4 * 24 * 40 * 40
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*4 * 24 * 40 * 40
        )
        self.convblock7 = nn.Sequential(
            # state size: ndf*4 * 24 * 40 * 40
            nn.Conv3d(ndf*4, ndf * 8, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*8 * 12 * 20 * 20
        )
        self.convblock8 = nn.Sequential(
            # state size: ndf*8 *  12 * 20 * 20
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*8 *  12 * 20 * 20
        )
        self.convblock9 = nn.Sequential(
            # state size: ndf*8 *  12 * 20 * 20
            nn.Conv3d(ndf*8, 320, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 6 * 10 * 10
        )
        self.convblock10 = nn.Sequential(
            # state size: 320 * 6 * 10 * 10
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 6 * 10 * 10
        )

        self.bridgeblock1= nn.Sequential(
            # state size: 320 * 6 * 10 * 10
            nn.Conv3d(320, 320, 3, stride = (1,2,2), padding=1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 6 * 5 * 5
        )
        self.bridgeblock2 = nn.Sequential(
            # state size: 320 * 6 * 5 * 5
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            nn.ConvTranspose3d(320, 320, kernel_size=(1,2,2), stride = (1,2,2), bias=use_bias),
            # state size: 320 * 6 * 10 * 10
        )

        self.deconvblock1 = nn.Sequential(
            # state size: (cat : 320+320) *  12 * 10 * 10
            nn.Conv3d(640, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 *  12 * 10 * 10
        )
        self.deconvblock2 = nn.Sequential(
            # state size: 320 *  12 * 10 * 10
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            nn.ConvTranspose3d(320, ndf*8, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*8 * 24 * 20 * 20
        )
        self.deconvblock3 = nn.Sequential(
            # state size: (cat : ndf*8+ndf*8) * 24 * 20 * 20
            nn.Conv3d(ndf*8*2, ndf*8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*8),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*8 * 24 * 20 * 20
        )
        self.deconvblock4 = nn.Sequential(
            # state size: ndf*8 * 24 * 20 * 20
            nn.Conv3d(ndf*8, ndf*8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*8),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock4_T = nn.ConvTranspose3d(ndf*8, ndf*4, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*4 * 48 * 40 * 40

        self.deconvblock5 = nn.Sequential(
            # state size: (cat : ndf*4+ndf*4) * 48 * 40 * 40
            nn.Conv3d(ndf*4 * 2, ndf*4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*4),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*4 * 48 * 40 * 40
        )
        self.deconvblock6 = nn.Sequential(
            # state size: ndf*4 * 48 * 40 * 40
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock6_T = nn.ConvTranspose3d(ndf * 4, ndf * 2, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*2 * 96 * 80 * 80

        self.deconvblock7 = nn.Sequential(
            # state size: (cat : ndf*2+ndf*2) *    96 * 80 * 80
            nn.Conv3d(ndf * 2 * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 *    96 * 80 * 80
        )
        self.deconvblock8 = nn.Sequential(
            # state size: ndf*2 *    96 * 80 * 80
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock8_T = nn.ConvTranspose3d(ndf * 2, ndf, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf *  96 * 160 * 160

        self.deconvblock9 = nn.Sequential(
            # state size: (cat : ndf+ndf) *   96 * 160 * 160
            nn.Conv3d(ndf * 2, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf *   96 * 160 * 160
        )
        self.deconvblock10 = nn.Sequential(
            # state size: ndf *   96 * 160 * 160
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf *   96 * 160 * 160
        )

        self.outputblock = nn.Sequential(
            # state size: ndf *  96 * 160 * 160
            nn.Conv3d(ndf, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid()
            # state size: channel_dim * 96 * 160 * 160
        )

        self.outputblockddt = nn.Sequential(
            # state size: ndf *  96 * 160 * 160
            nn.Conv3d(ndf, int((channel_dim-1)*k + 1), 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid()
            # state size: channel_dim * 96 * 160 * 160
        )


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        # for now it only supports one GPU
            encoder1 = self.convblock1(input)
            encoder2 = self.convblock2(encoder1)
            encoder3 = self.convblock3(encoder2)
            encoder4 = self.convblock4(encoder3)
            encoder5 = self.convblock5(encoder4)
            encoder6 = self.convblock6(encoder5)
            encoder7 = self.convblock7(encoder6)
            encoder8 = self.convblock8(encoder7)
            encoder9 = self.convblock9(encoder8)
            encoder10 = self.convblock10(encoder9)

            bridge1 = self.bridgeblock1(encoder10)
            bridge2 = self.bridgeblock2(bridge1)

            skip1 = torch.cat([bridge2, encoder10], 1)
            decoder1 = self.deconvblock1(skip1)
            decoder2 = self.deconvblock2(decoder1)
            skip2 = torch.cat([decoder2, encoder8], 1)
            decoder3 = self.deconvblock3(skip2)
            decoder4_out = self.deconvblock4(decoder3)
            decoder4 = self.deconvblock4_T(decoder4_out)
            skip3 = torch.cat([decoder4, encoder6], 1)
            decoder5 = self.deconvblock5(skip3)
            decoder6_out = self.deconvblock6(decoder5)
            decoder6 = self.deconvblock6_T(decoder6_out)
            skip4 = torch.cat([decoder6, encoder4], 1)
            decoder7 = self.deconvblock7(skip4)
            decoder8_out = self.deconvblock8(decoder7)
            decoder8 = self.deconvblock8_T(decoder8_out)
            skip5 = torch.cat([decoder8, encoder2], 1)
            decoder9 = self.deconvblock9(skip5)
            decoder10 = self.deconvblock10(decoder9)

            output = self.outputblock(decoder10)
            ddt = self.outputblockddt(decoder10)

            return output, ddt

class net_Dense(nn.Module):
    def __init__(self, ngpu, channel_dim, ndf = 30, use_bias = False):
        super(net_Dense, self).__init__()
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            #state size: 1 * 96 * 160 * 160
            nn.Conv3d(1, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 96 * 160 * 160
        )
        self.convblock2 = nn.Sequential(
            # state size: ndf * 96 * 160 * 160
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 96 * 160 * 160
        )
        self.convblock3 = nn.Sequential(
            # state size: ndf * 96 * 160 * 160
            nn.Conv3d(ndf + 4, ndf * 2, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 48 * 80 * 80
        )
        self.convblock4 = nn.Sequential(
            # state size: ndf*2 * 48 * 80 * 80
            nn.Conv3d(ndf * 2 + 4*2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 48 * 80 * 80
        )
        self.convblock5 = nn.Sequential(
            # state size: ndf * 48 * 80 * 80
            nn.Conv3d(ndf*2 + 4*3, ndf * 4, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 24 * 40 * 40
        )
        self.convblock6 = nn.Sequential(
            # state size: ndf*2 * 24 * 40 * 40
            nn.Conv3d(ndf * 4 + 4*4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 24 * 40 * 40
        )

        self.bridgeblock1= nn.Sequential(
            # state size: 320 * 24 * 40 * 40
            nn.Conv3d(ndf * 4 + 4*5, ndf * 8, 3, stride = (2,2,2), padding=1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 12 * 20 * 20
        )
        self.bridgeblock2 = nn.Sequential(
            # state size: 320 * 12 * 20 * 20
            nn.Conv3d(ndf * 8 + 4*6, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True)
        )
        self.bridgeblock3 = nn.Sequential(
            nn.ConvTranspose3d(ndf * 8 + 4*7, ndf * 4, kernel_size=(2,2,2), stride = (2,2,2), bias=use_bias)
        )

        self.deconvblock5 = nn.Sequential(
            # state size: (cat : ndf*4+ndf*4) * 48 * 40 * 40
            nn.Conv3d(ndf*4 + 4*8, ndf*4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*4),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*4 * 48 * 40 * 40
        )
        self.deconvblock6 = nn.Sequential(
            # state size: ndf*4 * 48 * 40 * 40
            nn.Conv3d(ndf * 4 + 4*9, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock6_T = nn.ConvTranspose3d(ndf * 4, ndf * 2, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*2 * 96 * 80 * 80

        self.deconvblock7 = nn.Sequential(
            # state size: (cat : ndf*2+ndf*2) *    96 * 80 * 80
            nn.Conv3d(ndf * 2 + 4*10, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 *    96 * 80 * 80
        )
        self.deconvblock8 = nn.Sequential(
            # state size: ndf*2 *    96 * 80 * 80
            nn.Conv3d(ndf * 2 + 4*11, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock8_T = nn.ConvTranspose3d(ndf * 2, ndf, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf *  96 * 160 * 160

        self.deconvblock9 = nn.Sequential(
            # state size: (cat : ndf+ndf) *   96 * 160 * 160
            nn.Conv3d(ndf + 4*12, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf *   96 * 160 * 160
        )
        self.deconvblock10 = nn.Sequential(
            # state size: ndf *   96 * 160 * 160
            nn.Conv3d(ndf + 4*13, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf *   96 * 160 * 160
        )

        self.outputblock = nn.Sequential(
            # state size: ndf *  96 * 160 * 160
            nn.Conv3d(ndf, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid()
            # state size: channel_dim * 96 * 160 * 160
        )

        self.outputblock1 = nn.Sequential(
            # state size: ndf*2 * 48 * 80 * 80use_bias
            nn.Conv3d(ndf * 2, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False)
            # state size: channel_dim * 96 * 160 * 160
        )


        self.outputblock2 = nn.Sequential(
            # state size: ndf*4 * 24*40*40
            nn.Conv3d(ndf * 4, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=(4,4,4), mode='trilinear', align_corners=False)
            # state size: channel_dim * 96 * 160 * 160
        )



        self.outputblock3 = nn.Sequential(
            # state size: ndf*8 * 12 * 20* 20
            nn.Conv3d(ndf * 8, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=(8,8,8), mode='trilinear', align_corners=False)
            # state size: channel_dim * 96 * 160 * 160
        )

        self.c1 = nn.Sequential(nn.Conv3d(ndf, 4, 1, 1, 0, bias=use_bias))
        self.c2 = nn.Sequential(nn.Conv3d(ndf * 2, 4, 1, 1, 0, bias=use_bias))
        self.c4 = nn.Sequential(nn.Conv3d(ndf * 4, 4, 1, 1, 0, bias=use_bias))
        self.c8 = nn.Sequential(nn.Conv3d(ndf * 8, 4, 1, 1, 0, bias=use_bias))
        self.down2 = nn.Upsample(scale_factor=(0.5, 0.5, 0.5), mode='trilinear', align_corners=False)
        self.down4 = nn.Upsample(scale_factor=(0.25, 0.25, 0.25), mode='trilinear', align_corners=False)
        self.down8 = nn.Upsample(scale_factor=(0.125, 0.125, 0.125), mode='trilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False)
        self.up8 = nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        # for now it only supports one GPU
            encoder1 = self.convblock1(input)
            encoder1d = self.c1(encoder1)
            encoder2 = self.convblock2(encoder1)
            encoder2d = self.c1(encoder2)
            encoder3 = self.convblock3(torch.cat([encoder2, encoder1d], 1))
            encoder3d = self.c2(encoder3)
            encoder4 = self.convblock4(torch.cat([encoder3, self.down2(encoder2d), self.down2(encoder1d)], 1))
            encoder4d = self.c2(encoder4)
            encoder5 = self.convblock5(torch.cat([encoder4, encoder3d, self.down2(encoder2d), self.down2(encoder1d)], 1))
            encoder5d = self.c4(encoder5)
            encoder6 = self.convblock6(torch.cat([encoder5, self.down2(encoder4d), self.down2(encoder3d),self.down4(encoder2d), self.down4(encoder1d)], 1))
            encoder6d = self.c4(encoder6)

            bridge1 = self.bridgeblock1(torch.cat([encoder6,encoder5d, self.down2(encoder4d), self.down2(encoder3d),self.down4(encoder2d), self.down4(encoder1d)], 1))
            bridge1d = self.c8(bridge1)
            bridge2 = self.bridgeblock2(torch.cat([bridge1,self.down2(encoder6d),self.down2(encoder5d), self.down4(encoder4d), self.down4(encoder3d),self.down8(encoder2d), self.down8(encoder1d)], 1))
            bridge2d = self.c8(bridge2)
            output3 = self.outputblock3(bridge2)

            bridge3 = self.bridgeblock3(torch.cat([bridge2, bridge1d,self.down2(encoder6d),self.down2(encoder5d), self.down4(encoder4d), self.down4(encoder3d),self.down8(encoder2d), self.down8(encoder1d)], 1))
            bridge3d = self.c4(bridge3)

            decoder5 = self.deconvblock5(torch.cat([bridge3, self.up2(bridge2d), self.up2(bridge1d),encoder6d,encoder5d, self.down2(encoder4d), self.down2(encoder3d),self.down4(encoder2d), self.down4(encoder1d)], 1))
            decoder5d = self.c4(decoder5)
            decoder6_out = self.deconvblock6(torch.cat([decoder5, bridge3d, self.up2(bridge2d), self.up2(bridge1d),encoder6d,encoder5d, self.down2(encoder4d), self.down2(encoder3d),self.down4(encoder2d), self.down4(encoder1d)], 1))

            output2 = self.outputblock2(decoder6_out)

            decoder6 = self.deconvblock6_T(decoder6_out)
            decoder6d = self.c2(decoder6)
            decoder7 = self.deconvblock7(torch.cat([decoder6, self.up2(decoder5d), self.up2(bridge3d), self.up4(bridge2d), self.up4(bridge1d),self.up2(encoder6d),self.up2(encoder5d), encoder4d, encoder3d,self.down2(encoder2d), self.down2(encoder1d)], 1))
            decoder7d = self.c2(decoder7)
            decoder8_out = self.deconvblock8(torch.cat([decoder7,decoder6d, self.up2(decoder5d), self.up2(bridge3d), self.up4(bridge2d), self.up4(bridge1d),self.up2(encoder6d),self.up2(encoder5d), encoder4d, encoder3d,self.down2(encoder2d), self.down2(encoder1d)], 1))

            output1 = self.outputblock1(decoder8_out)

            decoder8 = self.deconvblock8_T(decoder8_out)
            decoder8d = self.c1(decoder8)
            decoder9 = self.deconvblock9(torch.cat([decoder8, self.up2(decoder7d),self.up2(decoder6d), self.up4(decoder5d), self.up4(bridge3d), self.up8(bridge2d), self.up8(bridge1d),self.up4(encoder6d),self.up4(encoder5d), self.up2(encoder4d), self.up2(encoder3d),encoder2d, encoder1d], 1))
            decoder10 = self.deconvblock10(torch.cat([decoder9, decoder8d, self.up2(decoder7d),self.up2(decoder6d), self.up4(decoder5d), self.up4(bridge3d), self.up8(bridge2d), self.up8(bridge1d),self.up4(encoder6d),self.up4(encoder5d), self.up2(encoder4d), self.up2(encoder3d),encoder2d, encoder1d], 1))

            output = self.outputblock(decoder10)

            return output, output1, output2, output3

class net_Dense3(nn.Module):
    def __init__(self, ngpu, channel_dim, ndf = 30, use_bias = False):
        super(net_Dense3, self).__init__()
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            #state size: 1 * 96 * 160 * 160
            nn.Conv3d(1, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 96 * 160 * 160
        )
        self.convblock2 = nn.Sequential(
            # state size: ndf * 96 * 160 * 160
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 96 * 160 * 160
        )
        self.convblock3 = nn.Sequential(
            # state size: ndf * 96 * 160 * 160
            nn.Conv3d(ndf + ndf, ndf * 2, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 48 * 80 * 80
        )
        self.convblock4 = nn.Sequential(
            # state size: ndf*2 * 48 * 80 * 80
            nn.Conv3d(ndf * 2 + ndf + ndf, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 48 * 80 * 80
        )

        self.bridgeblock1= nn.Sequential(
            # state size: 320 * 48 * 80 * 80
            nn.Conv3d(ndf * 2 + ndf + ndf + ndf * 2, ndf * 4, 3, stride = (2,2,2), padding=1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 24 * 40 * 40
        )
        self.bridgeblock2 = nn.Sequential(
            # state size: 320 *  24 * 40 * 40
            nn.Conv3d(ndf * 4 + ndf + ndf + ndf * 2 + ndf * 2, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True)
        )
        self.bridgeblock3 = nn.Sequential(
            nn.ConvTranspose3d(ndf * 4 + ndf + ndf + ndf * 2 + ndf * 2 + ndf * 4, ndf * 2, kernel_size=(2,2,2), stride = (2,2,2), bias=use_bias)
        )


        self.deconvblock7 = nn.Sequential(
            # state size: (cat : ndf*2+ndf*2) *    96 * 80 * 80
            nn.Conv3d(ndf * 2 + ndf * 4 + ndf + ndf + ndf * 2 + ndf * 2 + ndf * 4, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 *    96 * 80 * 80
        )
        self.deconvblock8 = nn.Sequential(
            # state size: ndf*2 *    96 * 80 * 80
            nn.Conv3d(ndf * 2 + ndf * 2 + ndf * 4 + ndf + ndf + ndf * 2 + ndf * 2 + ndf * 4, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock8_T = nn.ConvTranspose3d(ndf * 2, ndf, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf *  96 * 160 * 160

        self.deconvblock9 = nn.Sequential(
            # state size: (cat : ndf+ndf) *   96 * 160 * 160
            nn.Conv3d(ndf + ndf * 2 + ndf * 2 + ndf * 4 + ndf + ndf + ndf * 2 + ndf * 2 + ndf * 4, ndf * 2, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf *   96 * 160 * 160
        )
        self.deconvblock10 = nn.Sequential(
            # state size: ndf *   96 * 160 * 160
            nn.Conv3d(ndf + ndf + ndf * 2 + ndf * 2 + ndf * 4 + ndf + ndf + ndf * 2 + ndf * 2 + ndf * 4, ndf * 2, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf *   96 * 160 * 160
        )

        self.outputblock = nn.Sequential(
            # state size: ndf *  96 * 160 * 160
            nn.Conv3d(ndf, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid()
            # state size: channel_dim * 96 * 160 * 160
        )

        self.outputblock1 = nn.Sequential(
            # state size: ndf*2 * 48 * 80 * 80use_bias
            nn.Conv3d(ndf * 2, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False)
            # state size: channel_dim * 96 * 160 * 160
        )


        self.outputblock2 = nn.Sequential(
            # state size: ndf*4 * 24*40*40
            nn.Conv3d(ndf * 4, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=(4,4,4), mode='trilinear', align_corners=False)
            # state size: channel_dim * 96 * 160 * 160
        )

        self.down2 = nn.Upsample(scale_factor=(0.5, 0.5, 0.5), mode='trilinear', align_corners=False)
        self.down4 = nn.Upsample(scale_factor=(0.25, 0.25, 0.25), mode='trilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        # for now it only supports one GPU
            encoder1 = self.convblock1(input)
            encoder2 = self.convblock2(encoder1)
            encoder3 = self.convblock3(torch.cat([encoder2, encoder1], 1))
            encoder4 = self.convblock4(torch.cat([encoder3, self.down2(encoder2), self.down2(encoder1)], 1))

            bridge1 = self.bridgeblock1(torch.cat([encoder4, encoder3,self.down2(encoder2), self.down2(encoder1)], 1))
            bridge2 = self.bridgeblock2(torch.cat([bridge1, self.down2(encoder4), self.down2(encoder3),self.down4(encoder2), self.down4(encoder1)], 1))

            output3 = self.outputblock2(bridge2)

            bridge3 = self.bridgeblock3(torch.cat([bridge2, bridge1, self.down2(encoder4), self.down2(encoder3),self.down4(encoder2), self.down4(encoder1)], 1))

            output2 = self.outputblock1(bridge3)

            decoder7 = self.deconvblock7(torch.cat([bridge3, self.up2(bridge2), self.up2(bridge1), encoder4, encoder3,self.down2(encoder2), self.down2(encoder1)], 1))
            decoder8_out = self.deconvblock8(torch.cat([decoder7,bridge3, self.up2(bridge2), self.up2(bridge1), encoder4, encoder3,self.down2(encoder2), self.down2(encoder1)], 1))

            output1 = self.outputblock1(decoder8_out)

            decoder8 = self.deconvblock8_T(decoder8_out)
            decoder9 = self.deconvblock9(torch.cat([decoder8, self.up2(decoder7),self.up2(bridge3), self.up4(bridge2), self.up4(bridge1), self.up2(encoder4), self.up2(encoder3),encoder2, encoder1], 1))
            decoder10 = self.deconvblock10(torch.cat([decoder9, decoder8, self.up2(decoder7),self.up2(bridge3), self.up4(bridge2), self.up4(bridge1), self.up2(encoder4), self.up2(encoder3),encoder2, encoder1], 1))

            output = self.outputblock(decoder10)

            return output, output1, output2, output3

class net_dist(nn.Module):
    def __init__(self, ngpu, channel_dim, ndf = 30, use_bias = False):
        super(net_dist, self).__init__()
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            #state size: 1 * 96 * 160 * 160
            nn.Conv3d(1, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 96 * 160 * 160
        )
        self.convblock2 = nn.Sequential(
            # state size: ndf * 96 * 160 * 160
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 96 * 160 * 160
        )
        self.convblock3 = nn.Sequential(
            # state size: ndf * 96 * 160 * 160
            nn.Conv3d(ndf, ndf * 2, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 48 * 80 * 80
        )
        self.convblock4 = nn.Sequential(
            # state size: ndf*2 * 48 * 80 * 80
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 48 * 80 * 80
        )
        self.convblock5 = nn.Sequential(
            # state size: ndf*2 * 48 * 80 * 80
            nn.Conv3d(ndf*2, ndf * 4, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*4 * 24 * 40 * 40
        )
        self.convblock6 = nn.Sequential(
            # state size: ndf*4 * 24 * 40 * 40
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*4 * 24 * 40 * 40
        )
        self.convblock7 = nn.Sequential(
            # state size: ndf*4 * 24 * 40 * 40
            nn.Conv3d(ndf*4, ndf * 8, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*8 * 12 * 20 * 20
        )
        self.convblock8 = nn.Sequential(
            # state size: ndf*8 *  12 * 20 * 20
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*8 *  12 * 20 * 20
        )
        self.convblock9 = nn.Sequential(
            # state size: ndf*8 *  12 * 20 * 20
            nn.Conv3d(ndf*8, 320, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 6 * 10 * 10
        )
        self.convblock10 = nn.Sequential(
            # state size: 320 * 6 * 10 * 10
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 6 * 10 * 10
        )

        self.bridgeblock1= nn.Sequential(
            # state size: 320 * 6 * 10 * 10
            nn.Conv3d(320, 320, 3, stride = (1,2,2), padding=1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 6 * 5 * 5
        )
        self.bridgeblock2 = nn.Sequential(
            # state size: 320 * 6 * 5 * 5
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            nn.ConvTranspose3d(320, 320, kernel_size=(1,2,2), stride = (1,2,2), bias=use_bias),
            # state size: 320 * 6 * 10 * 10
        )

        self.deconvblock1 = nn.Sequential(
            # state size: (cat : 320+320) *  12 * 10 * 10
            nn.Conv3d(640, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 *  12 * 10 * 10
        )
        self.deconvblock2 = nn.Sequential(
            # state size: 320 *  12 * 10 * 10
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            nn.ConvTranspose3d(320, ndf*8, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*8 * 24 * 20 * 20
        )
        self.deconvblock3 = nn.Sequential(
            # state size: (cat : ndf*8+ndf*8) * 24 * 20 * 20
            nn.Conv3d(ndf*8*2, ndf*8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*8),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*8 * 24 * 20 * 20
        )
        self.deconvblock4 = nn.Sequential(
            # state size: ndf*8 * 24 * 20 * 20
            nn.Conv3d(ndf*8, ndf*8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*8),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock4_T = nn.ConvTranspose3d(ndf*8, ndf*4, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*4 * 48 * 40 * 40

        self.deconvblock5 = nn.Sequential(
            # state size: (cat : ndf*4+ndf*4) * 48 * 40 * 40
            nn.Conv3d(ndf*4 * 2, ndf*4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*4),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*4 * 48 * 40 * 40
        )
        self.deconvblock6 = nn.Sequential(
            # state size: ndf*4 * 48 * 40 * 40
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock6_T = nn.ConvTranspose3d(ndf * 4, ndf * 2, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*2 * 96 * 80 * 80

        self.deconvblock7 = nn.Sequential(
            # state size: (cat : ndf*2+ndf*2) *    96 * 80 * 80
            nn.Conv3d(ndf * 2 * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 *    96 * 80 * 80
        )
        self.deconvblock8 = nn.Sequential(
            # state size: ndf*2 *    96 * 80 * 80
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock8_T = nn.ConvTranspose3d(ndf * 2, ndf, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf *  96 * 160 * 160

        self.deconvblock9 = nn.Sequential(
            # state size: (cat : ndf+ndf) *   96 * 160 * 160
            nn.Conv3d(ndf * 2, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf *   96 * 160 * 160
        )
        self.deconvblock10 = nn.Sequential(
            # state size: ndf *   96 * 160 * 160
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf *   96 * 160 * 160
        )

        self.outputblock = nn.Sequential(
            # state size: ndf *  96 * 160 * 160
            nn.Conv3d(ndf, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid()
            # state size: channel_dim * 96 * 160 * 160
        )

        self.outputblockdist = nn.Sequential(
            # state size: ndf *  96 * 160 * 160
            nn.Conv3d(ndf, channel_dim-1, 1, 1, 0, bias=use_bias),
        )


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose3d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        # for now it only supports one GPU
            encoder1 = self.convblock1(input)
            encoder2 = self.convblock2(encoder1)
            encoder3 = self.convblock3(encoder2)
            encoder4 = self.convblock4(encoder3)
            encoder5 = self.convblock5(encoder4)
            encoder6 = self.convblock6(encoder5)
            encoder7 = self.convblock7(encoder6)
            encoder8 = self.convblock8(encoder7)
            encoder9 = self.convblock9(encoder8)
            encoder10 = self.convblock10(encoder9)

            bridge1 = self.bridgeblock1(encoder10)
            bridge2 = self.bridgeblock2(bridge1)

            skip1 = torch.cat([bridge2, encoder10], 1)
            decoder1 = self.deconvblock1(skip1)
            decoder2 = self.deconvblock2(decoder1)
            skip2 = torch.cat([decoder2, encoder8], 1)
            decoder3 = self.deconvblock3(skip2)
            decoder4_out = self.deconvblock4(decoder3)
            decoder4 = self.deconvblock4_T(decoder4_out)
            skip3 = torch.cat([decoder4, encoder6], 1)
            decoder5 = self.deconvblock5(skip3)
            decoder6_out = self.deconvblock6(decoder5)
            decoder6 = self.deconvblock6_T(decoder6_out)
            skip4 = torch.cat([decoder6, encoder4], 1)
            decoder7 = self.deconvblock7(skip4)
            decoder8_out = self.deconvblock8(decoder7)
            decoder8 = self.deconvblock8_T(decoder8_out)
            skip5 = torch.cat([decoder8, encoder2], 1)
            decoder9 = self.deconvblock9(skip5)
            decoder10 = self.deconvblock10(decoder9)

            output = self.outputblock(decoder10)
            dist = self.outputblockdist(decoder10)

            return output, dist
