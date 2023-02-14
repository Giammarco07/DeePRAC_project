import torch.nn as nn
import torch

class net(nn.Module):
    def __init__(self, ngpu, channel_dim, ndf = 24, nbf = 240, use_bias = False):
        super(net, self).__init__()
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            #state size: 1 * 128 * 128 * 128
            nn.Conv3d(1, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 128 * 128 * 128
        )
        self.convblock2 = nn.Sequential(
            # state size: ndf * 128 * 128 * 128
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            # state size: ndf * 128 * 128 * 128
        )

        self.convblock3 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            # state size: ndf * 128 * 128 * 128
            nn.Conv3d(ndf, ndf * 2, 3, stride = (2,2,2), padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 64 * 64 * 64
        )
        self.convblock4 = nn.Sequential(
            # state size: ndf*2 * 64 * 64 * 64
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            # state size: ndf*2 * 64 * 64 * 64
        )
        self.convblock5 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            # state size: ndf*2 * 64 * 64 * 64
            nn.Conv3d(ndf*2, ndf * 4, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*4 * 32 * 32 * 32
        )

        self.convblock6 = nn.Sequential(
            # state size: ndf*4 * 32 * 32 * 32
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            # state size: ndf*4 * 32 * 32 * 32
        )
        self.convblock7 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            # state size: ndf*4 * 32 * 32 * 32
            nn.Conv3d(ndf*4, ndf * 8, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*8 * 16 * 16 * 16
        )
        self.convblock8 = nn.Sequential(
            # state size: ndf*8 * 16 * 16 * 16
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            # state size: ndf*8 * 16 * 16 * 16
        )
        self.convblock9 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            # state size: ndf*8 * 16 * 16 * 16
            nn.Conv3d(ndf*8, nbf, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
            nn.LeakyReLU(inplace= True),
            # state size: nbf * 8 * 8 * 8
        )
        self.convblock10 = nn.Sequential(
            # state size: nbf * 8 * 8 * 8
            nn.Conv3d(nbf, nbf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(nbf, nbf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
            # state size: nbf * 8 * 8 * 8
        )

        self.bridgeblock1= nn.Sequential(
            nn.LeakyReLU(inplace=True),
            # state size: nbf * 8 * 8 * 8
            nn.Conv3d(nbf, nbf, 3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
            nn.LeakyReLU(inplace= True),
            # state size: nbf * 4 * 4 * 4
        )
        self.bridgeblock2 = nn.Sequential(
            # state size: nbf * 4 * 4 * 4
            nn.Conv3d(nbf, nbf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
            nn.LeakyReLU(inplace= True),
            nn.ConvTranspose3d(nbf,nbf, kernel_size=2, stride=2, bias=use_bias),
            # state size: nbf * 8 * 8 * 8
        )

        self.deconvblock1 = nn.Sequential(
            # state size: nbfx2 * 8 * 8 * 8
            nn.Conv3d(nbf * 2, nbf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
            nn.LeakyReLU(inplace=True)
            # state size: nbf * 8 * 8 * 8
        )

        self.deconvblock2 = nn.Sequential(
            # state size: nbf * 8 * 8 * 8
            nn.Conv3d(nbf, nbf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(nbf, nbf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
        )
        self.deconvblock2_T = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(nbf, ndf*8, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*8 * 16 * 16 * 16
        )

        self.deconvblock3 = nn.Sequential(
            # state size: ndf*8x2 * 16 * 16 * 16
            nn.Conv3d(ndf*8 * 2, ndf*8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*8),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*8 * 16 * 16 * 16
        )
        self.deconvblock4 = nn.Sequential(
            # state size: ndf*8 * 16 * 16 * 16
            nn.Conv3d(ndf*8, ndf*8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*8),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
        )
        self.deconvblock4_T = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(ndf*8, ndf*4, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*4 * 32 * 32 * 32
        )

        self.deconvblock5 = nn.Sequential(
            # state size:ndf*4x2 * 32 * 32 * 32
            nn.Conv3d(ndf * 4*2, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace=True),
            # state size: ndf*4 * 32 * 32 * 32
        )

        self.deconvblock6 = nn.Sequential(
            # state size: ndf*4 * 32 * 32 * 32
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
        )
        self.deconvblock6_T =  nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 4, ndf * 2, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*2 * 64 * 64 * 64
        )

        self.deconvblock7 = nn.Sequential(
            # state size:ndf * 2x2 * 64 * 64 * 64
            nn.Conv3d(ndf * 2*2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace=True),
            # state size: ndf * 2 * 64 * 64 * 64
        )

        self.deconvblock8 = nn.Sequential(
            # state size: ndf*2 *  64 * 64 * 64
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
        )
        self.deconvblock8_T = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 2, ndf, kernel_size=(2,2,2), stride=(2,2,2), bias=use_bias),
            # state size: ndf *  128 * 128 * 128
        )

        self.deconvblock9 = nn.Sequential(
            # state size:ndfx2 *  128 * 128 * 128
            nn.Conv3d(ndf*2, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace=True),
            # state size: ndf *  128 * 128 * 128
        )

        self.deconvblock10 = nn.Sequential(
            # state size: ndf *  128 * 128 * 128
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            # state size: ndf *  128 * 128 * 128
        )

        self.outputblock = nn.Sequential(
            nn.LeakyReLU(inplace=True),
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
        # resblock
        encoder2 = self.convblock2(encoder1) #resblock
        encoder2 = encoder2 + encoder1 #residual

        encoder3 = self.convblock3(encoder2)
        # resblock
        encoder4 = self.convblock4(encoder3) #resblock
        encoder4 = encoder4 + encoder3 #residual

        encoder5 = self.convblock5(encoder4)
        # resblock
        encoder6 = self.convblock6(encoder5) #resblock
        encoder6 = encoder6 + encoder5 #residual

        encoder7 = self.convblock7(encoder6)
        # resblock
        encoder8 = self.convblock8(encoder7)  # resblock
        encoder8 = encoder8 + encoder7  # residual

        encoder9 = self.convblock9(encoder8)
        # resblock
        encoder10 = self.convblock10(encoder9)  # resblock
        encoder10 = encoder10 + encoder9  # residual

        bridge1 = self.bridgeblock1(encoder10)
        bridge2 = self.bridgeblock2(bridge1)

        skip1 = torch.cat([bridge2, encoder10], 1)
        decoder1 = self.deconvblock1(skip1)
        decoder2_out = self.deconvblock2(decoder1)
        decoder2_out = decoder2_out + decoder1
        decoder2 = self.deconvblock2_T(decoder2_out)

        skip2 = torch.cat([decoder2, encoder8], 1)
        decoder3 = self.deconvblock3(skip2)
        decoder4_out = self.deconvblock4(decoder3)
        decoder4_out = decoder4_out + decoder3
        output3 = self.outputblock3(decoder4_out)
        decoder4 = self.deconvblock4_T(decoder4_out)

        skip3 = torch.cat([decoder4, encoder6], 1)
        decoder5 = self.deconvblock5(skip3)
        decoder6_out = self.deconvblock6(decoder5)
        decoder6_out = decoder6_out + decoder5
        output2 = self.outputblock2(decoder6_out)
        decoder6 = self.deconvblock6_T(decoder6_out)

        skip4 = torch.cat([decoder6, encoder4], 1)
        decoder7 = self.deconvblock7(skip4)
        decoder8_out = self.deconvblock8(decoder7)
        decoder8_out = decoder8_out + decoder7
        output1 = self.outputblock1(decoder8_out)
        decoder8 = self.deconvblock8_T(decoder8_out)

        skip5 = torch.cat([decoder8, encoder2], 1)
        decoder9 = self.deconvblock9(skip5)
        decoder10 = self.deconvblock10(decoder9)
        decoder10 = decoder10 + decoder9

        output = self.outputblock(decoder10)

        return output, output1, output2, output3