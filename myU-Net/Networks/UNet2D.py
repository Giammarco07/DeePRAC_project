import torch.nn as nn
import torch

class net(nn.Module):
    def __init__(self, ngpu, channel_dim, ndf = 32, use_bias = False):
        super(net, self).__init__()
        print('use bias in the network: ',use_bias)
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            #state size: 1 * 512 * 512
            nn.Conv2d(1, ndf, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size: ndf * 512 * 512
        )
        self.convblock2 = nn.Sequential(
            # state size: ndf * 512 * 512
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size: ndf * 512 * 512
        )
        self.convblock3 = nn.Sequential(
            # state size: ndf * 512 * 512
            nn.Conv2d(ndf, ndf * 2, 3, stride = (2,2), padding = 1, bias=use_bias),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # state size: ndf*2 * 256 * 256
        )
        self.convblock4 = nn.Sequential(
            # state size: ndf*2 * 256 * 256
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # state size: ndf*2 *  256 * 256
        )
        self.convblock5 = nn.Sequential(
            # state size: ndf*2 * 256 * 256
            nn.Conv2d(ndf*2, ndf * 4, 3, stride = 2, padding = 1, bias=use_bias),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            # state size: ndf*4 * 128 * 128
        )
        self.convblock6 = nn.Sequential(
            # state size: ndf*4 * 128 * 128
            nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            # state size: ndf*4 * 128 * 128
        )
        self.convblock7 = nn.Sequential(
            # state size: ndf*4 * 128 * 128
            nn.Conv2d(ndf*4, ndf * 8, 3, stride = 2, padding = 1, bias=use_bias),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size: ndf*8 * 64 * 64
        )
        self.convblock8 = nn.Sequential(
            # state size: ndf*8 * 64 * 64
            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size: ndf*8 * 64 * 64
        )
        self.convblock9 = nn.Sequential(
            # state size: ndf*8 * 64 * 64
            nn.Conv2d(ndf*8, 320, 3, stride = 2, padding = 1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            # state size: 320 * 32 * 32
        )
        self.convblock10 = nn.Sequential(
            # state size: 320 * 32 * 32
            nn.Conv2d(320, 320, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            # state size: 320 * 32 * 32
        )
        self.convblock11 = nn.Sequential(
            # state size: 320 * 32 * 32
            nn.Conv2d(320, 320, 3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            # state size: 320 * 16*16
        )
        self.convblock12 = nn.Sequential(
            # state size: 320 * 16*16
            nn.Conv2d(320, 320, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            # state size: 320 * 16*16
        )
        self.convblock13 = nn.Sequential(
            # state size: 320 * 16*16
            nn.Conv2d(320, 320, 3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            # state size: 320 * 8 * 8
        )
        self.convblock14 = nn.Sequential(
            # state size: 320 * 8*8
            nn.Conv2d(320, 320, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            # state size: 320 * 8*8
        )

        self.bridgeblock1= nn.Sequential(
            # state size: 320 * 8*8
            nn.Conv2d(320, 320, 3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            # state size: 320 * 4*4
        )
        self.bridgeblock2 = nn.Sequential(
            # state size: 320 * 4*4
            nn.Conv2d(320, 320, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            nn.ConvTranspose2d(320, 320, kernel_size=2, stride=2, bias=use_bias),
            # state size: 320 * 8*8
        )
        self.deconvblock1 = nn.Sequential(
            # state size: (cat : 320+320) * 8*8
            nn.Conv2d(640, 320, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            # state size: 320 * 8*8
        )
        self.deconvblock2 = nn.Sequential(
            # state size: 320 * 8*8
            nn.Conv2d(320, 320, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            nn.ConvTranspose2d(320, 320, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*8 * 16*16
        )
        self.deconvblock3 = nn.Sequential(
            # state size: (cat : 320+320) * 16*16
            nn.Conv2d(640, 320, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            # state size: 320 * 16*16
        )
        self.deconvblock4 = nn.Sequential(
            # state size: 320 * 16 * 16
            nn.Conv2d(320, 320, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            nn.ConvTranspose2d(320, 320, kernel_size=2, stride=2, bias=use_bias),
            # state size: 320 * 32*32
        )
        self.deconvblock5 = nn.Sequential(
            # state size: (cat : 320+320) * 32*32
            nn.Conv2d(640, 320, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            # state size: 320 * 32*32
        )
        self.deconvblock6 = nn.Sequential(
            # state size: 320 * 32*32
            nn.Conv2d(320, 320, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            nn.ConvTranspose2d(320, ndf*8, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*8 * 64*64
        )
        self.deconvblock7 = nn.Sequential(
            # state size: (cat : ndf*8+ndf*8) *  64*64
            nn.Conv2d(ndf*8*2, ndf*8, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf*8),
            nn.ReLU(True),
            # state size: ndf*8 * 64*64
        )
        self.deconvblock8 = nn.Sequential(
            # state size: ndf*8 * 64*64
            nn.Conv2d(ndf*8, ndf*8, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf*8, ndf*4, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*4 * 128*128
        )
        self.deconvblock9 = nn.Sequential(
            # state size: (cat : ndf*4+ndf*4) * 128*128
            nn.Conv2d(ndf*4 * 2, ndf*4, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf*4),
            nn.ReLU(True),
            # state size: ndf*4 * 128*128
        )
        self.deconvblock10 = nn.Sequential(
            # state size: ndf*4 * 128*128
            nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf*2 * 256*256
        )
        self.deconvblock11 = nn.Sequential(
            # state size: (cat : ndf*2+ndf*2) * 256*256
            nn.Conv2d(ndf * 2 * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # state size: ndf*2 * 256*256
        )
        self.deconvblock12 = nn.Sequential(
            # state size: ndf*2 * 256*256
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=2, stride=2, bias=use_bias),
            # state size: ndf *512*512
        )
        self.deconvblock13 = nn.Sequential(
            # state size: (cat : ndf+ndf) *  512*512
            nn.Conv2d(ndf * 2, ndf, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size: ndf *  512*512
        )
        self.deconvblock14 = nn.Sequential(
            # state size: ndf  * 512*512
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size: ndf * 512*512
        )

        self.outputblock = nn.Sequential(
            # state size: ndf * 512*512
            nn.Conv2d(ndf, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            #nn.Sigmoid()
            # state size: channel_dim * 512*512
        )

        self.outputblock1 = nn.Sequential(
            # state size: ndf*2 * 256*256
            nn.Conv2d(ndf*2, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            #nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            # state size: channel_dim * 256*256
        )

        self.outputblock2 = nn.Sequential(
            # state size: ndf*4 * 128*128
            nn.Conv2d(ndf*4, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            #nn.Sigmoid(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
            # state size: channel_dim * 128*128
        )

        self.outputblock3 = nn.Sequential(
            # state size: ndf*8 * 64*64
            nn.Conv2d(ndf*8, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            #nn.Sigmoid(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
            # state size: channel_dim * 64*64
        )

        self.outputblock4 = nn.Sequential(
            # state size: 320 * 32*32
            nn.Conv2d(320, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            #nn.Sigmoid(),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
            # state size: channel_dim * 32*32
        )

        self.outputblock5 = nn.Sequential(
            # state size: 320 * 16*16
            nn.Conv2d(320, channel_dim, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
            #nn.Sigmoid(),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
            # state size: channel_dim * 16*16
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                torch.manual_seed(786)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            '''
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.kaiming_normal_(m.weight.data, a=1, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            '''

    def forward(self, input):
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
            encoder11 = self.convblock11(encoder10)
            encoder12 = self.convblock12(encoder11)
            encoder13 = self.convblock13(encoder12)
            encoder14 = self.convblock14(encoder13)

            bridge1 = self.bridgeblock1(encoder14)
            bridge2 = self.bridgeblock2(bridge1)

            skip1 = torch.cat([bridge2, encoder14], 1)
            decoder1 = self.deconvblock1(skip1)
            decoder2 = self.deconvblock2(decoder1)
            skip2 = torch.cat([decoder2, encoder12], 1)
            decoder3 = self.deconvblock3(skip2)
            decoder4 = self.deconvblock4(decoder3)
            skip3 = torch.cat([decoder4, encoder10], 1)
            decoder5 = self.deconvblock5(skip3)
            decoder6 = self.deconvblock6(decoder5)
            skip4 = torch.cat([decoder6, encoder8], 1)
            decoder7 = self.deconvblock7(skip4)
            decoder8 = self.deconvblock8(decoder7)
            skip5 = torch.cat([decoder8, encoder6], 1)
            decoder9 = self.deconvblock9(skip5)
            decoder10 = self.deconvblock10(decoder9)
            skip6 = torch.cat([decoder10, encoder4], 1)
            decoder11 = self.deconvblock11(skip6)
            decoder12 = self.deconvblock12(decoder11)
            skip7 = torch.cat([decoder12, encoder2], 1)
            decoder13 = self.deconvblock13(skip7)
            decoder14 = self.deconvblock14(decoder13)

            output = self.outputblock(decoder14)
            output1 = self.outputblock1(decoder11)
            output2 = self.outputblock2(decoder9)
            output3 = self.outputblock3(decoder7)
            output4 = self.outputblock4(decoder5)
            output5 = self.outputblock5(decoder3)


            return output, output1, output2, output3, output4, output5