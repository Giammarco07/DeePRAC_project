import torch.nn as nn
import torch.nn.functional as F
import torch

class net(nn.Module):
    def __init__(self, ngpu, channel_dim, ndf = 24, use_bias=True):
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
            nn.Conv3d(ndf*8, 320, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 8 * 8 * 8
        )
        self.convblock10 = nn.Sequential(
            # state size: 320 * 8 * 8 * 8
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            # state size: 320 * 8 * 8 * 8
        )

        self.bridgeblock1= nn.Sequential(
            nn.LeakyReLU(inplace=True),
            # state size: 320 * 8 * 8 * 8
            nn.Conv3d(320, 320, 3, stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            # state size: 320 * 4 * 4 * 4
        )
        self.bridgeblock2 = nn.Sequential(
            # state size: 320 * 4 * 4 * 4
            nn.Conv3d(320, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            nn.ConvTranspose3d(320, 320, kernel_size=2, stride=2, bias=use_bias),
            # state size: 320 * 8 * 8 * 8
        )

        self.deconvblock2 = nn.Sequential(
            # state size: 320 * 8 * 8 * 8
            nn.Conv3d(320*2, 320, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(inplace= True),
            nn.ConvTranspose3d(320, ndf*8, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*8 * 16 * 16 * 16
        )
        self.deconvblock4 = nn.Sequential(
            # state size: ndf*8 * 16 * 16 * 16
            nn.Conv3d(ndf*8*2, ndf*8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*8),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock4_T = nn.ConvTranspose3d(ndf*8, ndf*4, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*4 * 32 * 32 * 32


        self.deconvblock6 = nn.Sequential(
            # state size: ndf*4 * 32 * 32 * 32
            nn.Conv3d(ndf * 4*2, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock6_T = nn.ConvTranspose3d(ndf * 4, ndf * 2, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*2 * 64 * 64 * 64

        self.deconvblock8 = nn.Sequential(
            # state size: ndf*2 *  64 * 64 * 64
            nn.Conv3d(ndf * 2*2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock8_T = nn.ConvTranspose3d(ndf * 2, ndf, kernel_size=(2,2,2), stride=(2,2,2), bias=use_bias)
            # state size: ndf *  128 * 128 * 128


        self.deconvblock10 = nn.Sequential(
            # state size: ndf *  128 * 128 * 128
            nn.Conv3d(ndf*2, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
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
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=use_bias)
            # state size: channel_dim * 128 * 128 * 128
        )

        self.outputblock2 = nn.Sequential(
            # state size: ndf*4 * 32*32*32
            nn.Conv3d(ndf * 4, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=use_bias)
            # state size: channel_dim * 128 * 128 * 128
        )

        self.outputblock3 = nn.Sequential(
            # state size: ndf*8 * 16*16*16
            nn.Conv3d(ndf * 8, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=use_bias)
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
            encoder2 = self.convblock2(encoder1) #resblock
            encoder21 = encoder2 + encoder1 #residual

            encoder3 = self.convblock3(encoder21)
            encoder4 = self.convblock4(encoder3) #resblock
            encoder43 = encoder4 + encoder3 #residual

            encoder5 = self.convblock5(encoder43)
            encoder6 = self.convblock6(encoder5) #resblock
            encoder65 = encoder6 + encoder5 #residual

            encoder7 = self.convblock7(encoder65)
            encoder8 = self.convblock8(encoder7) #resblock
            encoder87 = encoder8 + encoder7 #residual

            encoder9 = self.convblock9(encoder87)
            encoder10 = self.convblock10(encoder9) #resblock
            encoder109 = encoder10 + encoder9

            bridge1 = self.bridgeblock1(encoder109)
            bridge2 = self.bridgeblock2(bridge1)

            skip1 = torch.cat([bridge2, encoder10], 1)
            decoder2 = self.deconvblock2(skip1)
            skip2 = torch.cat([decoder2, encoder8], 1)
            decoder4_out = self.deconvblock4(skip2)
            output3 = self.outputblock3(decoder4_out)
            decoder4 = self.deconvblock4_T(decoder4_out)
            skip3 = torch.cat([decoder4, encoder6], 1)
            decoder6_out = self.deconvblock6(skip3)
            output2 = self.outputblock2(decoder6_out)
            decoder6 = self.deconvblock6_T(decoder6_out)
            skip4 = torch.cat([decoder6, encoder4], 1)
            decoder8_out = self.deconvblock8(skip4)
            output1 = self.outputblock1(decoder8_out)
            decoder8 = self.deconvblock8_T(decoder8_out)
            skip5 = torch.cat([decoder8, encoder2], 1)
            decoder10 = self.deconvblock10(skip5)

            output = self.outputblock(decoder10)

            return output, output1, output2, output3


class Deep_net(nn.Module):
    def __init__(self, ngpu, channel_dim, ndf=30, nbf = 240, use_bias=True):
        super(Deep_net, self).__init__()
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
        self.convblock6_bis = nn.Sequential(
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
        self.convblock8_bis = nn.Sequential(
            # state size: ndf*8 * 16 * 16 * 16
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            # state size: ndf*8 * 16 * 16 * 16
        )
        self.convblock8_tris = nn.Sequential(
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
        self.convblock10_bis = nn.Sequential(
            # state size: nbf * 8 * 8 * 8
            nn.Conv3d(nbf, nbf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(nbf, nbf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
            # state size: nbf * 8 * 8 * 8
        )
        self.convblock10_tris = nn.Sequential(
            # state size: nbf * 8 * 8 * 8
            nn.Conv3d(nbf, nbf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(nbf, nbf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
            # state size: nbf * 8 * 8 * 8
        )
        self.convblock10_final = nn.Sequential(
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
            nn.ConvTranspose3d(nbf, nbf, kernel_size=2, stride=2, bias=use_bias),
            # state size: nbf * 8 * 8 * 8
        )

        self.deconvblock2 = nn.Sequential(
            # state size: nbf * 8 * 8 * 8
            nn.Conv3d(nbf*2, nbf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(nbf),
            nn.LeakyReLU(inplace= True),
            nn.ConvTranspose3d(nbf, ndf*8, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*8 * 16 * 16 * 16
        )
        self.deconvblock4 = nn.Sequential(
            # state size: ndf*8 * 16 * 16 * 16
            nn.Conv3d(ndf*8*2, ndf*8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*8),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock4_T = nn.ConvTranspose3d(ndf*8, ndf*4, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*4 * 32 * 32 * 32


        self.deconvblock6 = nn.Sequential(
            # state size: ndf*4 * 32 * 32 * 32
            nn.Conv3d(ndf * 4*2, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock6_T = nn.ConvTranspose3d(ndf * 4, ndf * 2, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*2 * 64 * 64 * 64

        self.deconvblock8 = nn.Sequential(
            # state size: ndf*2 *  64 * 64 * 64
            nn.Conv3d(ndf * 2*2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock8_T = nn.ConvTranspose3d(ndf * 2, ndf, kernel_size=(2,2,2), stride=(2,2,2), bias=use_bias)
            # state size: ndf *  128 * 128 * 128


        self.deconvblock10 = nn.Sequential(
            # state size: ndf *  128 * 128 * 128
            nn.Conv3d(ndf*2, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
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
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=use_bias)
            # state size: channel_dim * 128 * 128 * 128
        )

        self.outputblock2 = nn.Sequential(
            # state size: ndf*4 * 32*32*32
            nn.Conv3d(ndf * 4, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=use_bias)
            # state size: channel_dim * 128 * 128 * 128
        )

        self.outputblock3 = nn.Sequential(
            # state size: ndf*8 * 16*16*16
            nn.Conv3d(ndf * 8, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=use_bias)
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
            # resblock x2
            encoder6 = self.convblock6(encoder5) #resblock
            encoder6 = F.leaky_relu(encoder6 + encoder5) #residual
            encoder6x2 = self.convblock6_bis(encoder6) #resblock
            encoder6x2 = encoder6x2 + encoder6 #residual

            encoder7 = self.convblock7(encoder6x2)
            # resblock x 3
            encoder8 = self.convblock8(encoder7) #resblock
            encoder8 = F.leaky_relu(encoder8 + encoder7) #residual
            encoder8x2 = self.convblock8_bis(encoder8)  # resblock
            encoder8x2 = F.leaky_relu(encoder8x2 + encoder8) #residual
            encoder8x3 = self.convblock8_tris(encoder8x2)  # resblock
            encoder8x3 = encoder8x3 + encoder8x2

            encoder9 = self.convblock9(encoder8x3)
            #resblock x 4
            encoder10 = self.convblock10(encoder9) #resblock
            encoder10 = F.leaky_relu(encoder10 + encoder9) #residual
            encoder10x2 = self.convblock10_bis(encoder10)  # resblock
            encoder10x2 = F.leaky_relu(encoder10x2 + encoder10) #residual
            encoder10x3 = self.convblock10_tris(encoder10x2)  # resblock
            encoder10x3= F.leaky_relu(encoder10x3 + encoder10x2) #residual
            encoder10x4 = self.convblock10_final(encoder10x3)  # resblock
            encoder10x4 = encoder10x4 + encoder10x3

            bridge1 = self.bridgeblock1(encoder10x4)
            bridge2 = self.bridgeblock2(bridge1)

            skip1 = torch.cat([bridge2, encoder10], 1)
            decoder2 = self.deconvblock2(skip1)
            skip2 = torch.cat([decoder2, encoder8], 1)
            decoder4_out = self.deconvblock4(skip2)
            output3 = self.outputblock3(decoder4_out)
            decoder4 = self.deconvblock4_T(decoder4_out)
            skip3 = torch.cat([decoder4, encoder6], 1)
            decoder6_out = self.deconvblock6(skip3)
            output2 = self.outputblock2(decoder6_out)
            decoder6 = self.deconvblock6_T(decoder6_out)
            skip4 = torch.cat([decoder6, encoder4], 1)
            decoder8_out = self.deconvblock8(skip4)
            output1 = self.outputblock1(decoder8_out)
            decoder8 = self.deconvblock8_T(decoder8_out)
            skip5 = torch.cat([decoder8, encoder2], 1)
            decoder10 = self.deconvblock10(skip5)

            output = self.outputblock(decoder10)

            return output, output1, output2, output3
            
class Deep_net_64(nn.Module):
    def __init__(self, ngpu, channel_dim, ndf=30, use_bias=True):
        super(Deep_net_64, self).__init__()
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            #state size: 1 * 32 64 64
            nn.Conv3d(1, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 32 64 64
        )
        self.convblock2 = nn.Sequential(
            # state size: ndf * 32 64 64
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            # state size: ndf * 32 64 64
        )

        self.convblock3 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            # state size: ndf * 32 64 64
            nn.Conv3d(ndf, ndf * 2, 3, stride = (2,2,2), padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*2 * 16 32 32
        )
        self.convblock4 = nn.Sequential(
            # state size: ndf*2 * 16 32 32
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            # state size: ndf*2 *16 32 32
        )
        self.convblock5 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            # state size: ndf*2 * 16 32 32
            nn.Conv3d(ndf*2, ndf * 4, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*4 * 8 16 16
        )

        self.convblock6 = nn.Sequential(
            # state size: ndf*4 * 8 16 16
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            # state size: ndf*4 * 8 16 16
        )
        self.convblock6_bis = nn.Sequential(
            # state size: ndf*4 * 8 16 16
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            # state size: ndf*4 * 8 16 16
        )
        self.convblock7 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            # state size: ndf*4 * 8 16 16
            nn.Conv3d(ndf*4, ndf * 8, 3, stride = 2, padding = 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            # state size: ndf*8 * 4 8 8
        )
        self.convblock8 = nn.Sequential(
            # state size: ndf*8 * 4 8 8
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            # state size: ndf*8 * 4 8 8
        )
        self.convblock8_bis = nn.Sequential(
            # state size: ndf*8 * 4 8 8
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            # state size: ndf*8 * 4 8 8
        )
        self.convblock8_tris = nn.Sequential(
            # state size: ndf*8 * 4 8 8
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            # state size: ndf*8 * 4 8 8
        )

        self.bridgeblock1= nn.Sequential(
            nn.LeakyReLU(inplace=True),
            # state size: ndf * 8 * 4 * 8 * 8
            nn.Conv3d(ndf * 8, ndf * 8, 3, stride=(1,2,2), padding=1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            # state size: ndf * 8 * 4 * 4 * 4
        )
        self.bridgeblock2 = nn.Sequential(
            # state size: ndf * 8* 4 * 4 * 4
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(inplace= True),
            nn.ConvTranspose3d(ndf * 8, ndf * 8, kernel_size=(1,2,2), stride=(1,2,2), bias=use_bias),
            # state size: ndf * 8 * 4 * 8 * 8
        )

        self.deconvblock4 = nn.Sequential(
            # state size: ndf*8 * 4 * 8 * 8
            nn.Conv3d(ndf*8*2, ndf*8, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf*8),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock4_T = nn.ConvTranspose3d(ndf*8, ndf*4, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*4 * 8 16 16


        self.deconvblock6 = nn.Sequential(
            # state size: ndf*4 * 8 16 16
            nn.Conv3d(ndf * 4*2, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock6_T = nn.ConvTranspose3d(ndf * 4, ndf * 2, kernel_size=2, stride=2, bias=use_bias)
            # state size: ndf*2 * 16 32 32

        self.deconvblock8 = nn.Sequential(
            # state size: ndf*2 *  16 32 32
            nn.Conv3d(ndf * 2*2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(inplace= True),)
        self.deconvblock8_T = nn.ConvTranspose3d(ndf * 2, ndf, kernel_size=(2,2,2), stride=(2,2,2), bias=use_bias)
            # state size: ndf *  32 64 64


        self.deconvblock10 = nn.Sequential(
            # state size: ndf *  32 64 64
            nn.Conv3d(ndf*2, ndf, 3, 1, 1, bias=use_bias),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(inplace= True),
            # state size: ndf *  32 64 64
        )

        self.outputblock = nn.Sequential(
            # state size: ndf * 32 64 64
            nn.Conv3d(ndf, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid()
            # state size: channel_dim * 32 64 64
        )

        self.outputblock1 = nn.Sequential(
            # state size: ndf*2 * 16 32 32
            nn.Conv3d(ndf * 2, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=use_bias)
            # state size: channel_dim * 32 64 64
        )

        self.outputblock2 = nn.Sequential(
            # state size: ndf*4 * 8 16 16
            nn.Conv3d(ndf * 4, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=use_bias)
            # state size: channel_dim * 32 64 64
        )

        self.outputblock3 = nn.Sequential(
            # state size: ndf*8 * 4 8 8
            nn.Conv3d(ndf * 8, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=use_bias)
            # state size: channel_dim * 32 64 64
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
            # resblock x2
            encoder6 = self.convblock6(encoder5) #resblock
            encoder6 = F.leaky_relu(encoder6 + encoder5) #residual
            encoder6x2 = self.convblock6_bis(encoder6) #resblock
            encoder6x2 = encoder6x2 + encoder6 #residual

            encoder7 = self.convblock7(encoder6x2)
            # resblock x 3
            encoder8 = self.convblock8(encoder7) #resblock
            encoder8 = F.leaky_relu(encoder8 + encoder7) #residual
            encoder8x2 = self.convblock8_bis(encoder8)  # resblock
            encoder8x2 = F.leaky_relu(encoder8x2 + encoder8) #residual
            encoder8x3 = self.convblock8_tris(encoder8x2)  # resblock
            encoder8x3 = encoder8x3 + encoder8x2

            bridge1 = self.bridgeblock1(encoder8x3)
            bridge2 = self.bridgeblock2(bridge1)

            skip1 = torch.cat([bridge2, encoder8], 1)
            decoder4_out = self.deconvblock4(skip1)
            output3 = self.outputblock3(decoder4_out)
            decoder4 = self.deconvblock4_T(decoder4_out)
            skip3 = torch.cat([decoder4, encoder6], 1)
            decoder6_out = self.deconvblock6(skip3)
            output2 = self.outputblock2(decoder6_out)
            decoder6 = self.deconvblock6_T(decoder6_out)
            skip4 = torch.cat([decoder6, encoder4], 1)
            decoder8_out = self.deconvblock8(skip4)
            output1 = self.outputblock1(decoder8_out)
            decoder8 = self.deconvblock8_T(decoder8_out)
            skip5 = torch.cat([decoder8, encoder2], 1)
            decoder10 = self.deconvblock10(skip5)

            output = self.outputblock(decoder10)

            return output, output1, output2, output3
