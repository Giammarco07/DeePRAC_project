import torch.nn as nn
import torch

class net(nn.Module):
    def __init__(self, ngpu, ndf = 8, use_bias = True, in_c=1):
        super(net, self).__init__()
        self.ngpu = ngpu

        #base encoder
        self.convblock1 = nn.Sequential(
            nn.Conv3d(in_c, ndf, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace= True),
        )
        self.mp1 = nn.MaxPool3d(2, stride=2)
        self.convblock2 = nn.Sequential(
            nn.Conv3d(ndf, ndf*2, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf*2),
            nn.ReLU(inplace= True),
        )
        self.mp2 = nn.MaxPool3d(2, stride=2)
        self.convblock3 = nn.Sequential(
            nn.Conv3d(ndf*2, ndf*4, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf*4),
            nn.ReLU(inplace= True),
        )
        self.mp3 = nn.MaxPool3d(2, stride=2)
        self.convblock4 = nn.Sequential(
            nn.Conv3d(ndf*4, ndf*8, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf*8),
            nn.ReLU(inplace= True),
        )
        self.mp4 = nn.MaxPool3d(2, stride=2)
        self.convblock5 = nn.Sequential(
            nn.Conv3d(ndf*8, ndf*16, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf*16),
            nn.ReLU(inplace= True),
        )

        #vessel extraction decoder
        self.bridgeblock1 = nn.Sequential(
            nn.Conv3d(ndf*16, ndf*32, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf*32),
            nn.ReLU(inplace= True),
            nn.Conv3d(ndf * 32, ndf*8, 1, 1, 0, bias=use_bias),
            nn.ConvTranspose3d(ndf*8, ndf * 8, kernel_size=4, padding=1,stride=2, bias=use_bias),
        )
        self.deconvblock11 = nn.Sequential(
            nn.Conv3d(ndf * 8 + ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 8, ndf * 8, kernel_size=4, padding=1, stride=2,bias=use_bias),
        )
        self.deconvblock12 = nn.Sequential(
            nn.Conv3d(ndf * 8 + ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 4, ndf * 4, kernel_size=4, padding=1, stride=2,bias=use_bias),
        )
        self.deconvblock13 = nn.Sequential(
            nn.Conv3d(ndf * 4 + ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 2, ndf * 2, kernel_size=4, padding=1, stride=2,bias=use_bias),
        )
        self.outputblock1 = nn.Sequential(
            nn.Conv3d(ndf * 2 + ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, 2, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
        )

        #centerness decoder
        self.bridgeblock2 = nn.Sequential(
            nn.Conv3d(ndf * 16, ndf * 32, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 32),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf * 32, ndf * 8, 1, 1, 0, bias=use_bias),
            nn.ConvTranspose3d(ndf * 8, ndf * 8, kernel_size=4, padding=1, stride=2, bias=use_bias),
        )
        self.deconvblock21 = nn.Sequential(
            nn.Conv3d(ndf * 8 + ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 8, ndf * 8, kernel_size=4, padding=1, stride=2, bias=use_bias),
        )
        self.deconvblock22 = nn.Sequential(
            nn.Conv3d(ndf * 8 + ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 4, ndf * 4, kernel_size=4, padding=1, stride=2, bias=use_bias),
        )
        self.deconvblock23 = nn.Sequential(
            nn.Conv3d(ndf * 4 + ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 2, ndf * 2, kernel_size=4, padding=1, stride=2, bias=use_bias),
        )
        self.outputblock2 = nn.Sequential(
            nn.Conv3d(ndf * 2 + ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf,1, 1, 1, 0, bias=use_bias),
            nn.ReLU(inplace=True),
        )

        #topological distance decoder
        self.bridgeblock3 = nn.Sequential(
            nn.Conv3d(ndf * 16, ndf * 32, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 32),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf * 32, ndf * 8, 1, 1, 0, bias=use_bias),
            nn.ConvTranspose3d(ndf * 8, ndf * 8, kernel_size=4, padding=1, stride=2, bias=use_bias),
        )
        self.deconvblock31 = nn.Sequential(
            nn.Conv3d(ndf * 8 + ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 8, ndf * 8, kernel_size=4, padding=1, stride=2, bias=use_bias),
        )
        self.deconvblock32 = nn.Sequential(
            nn.Conv3d(ndf * 8 + ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 4, ndf * 4, kernel_size=4, padding=1, stride=2, bias=use_bias),
        )
        self.deconvblock33 = nn.Sequential(
            nn.Conv3d(ndf * 4 + ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 2, ndf * 2, kernel_size=4, padding=1, stride=2, bias=use_bias),
        )
        self.outputblock3 = nn.Sequential(
            nn.Conv3d(ndf * 2 + ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf,8, 1, 1, 0, bias=use_bias),
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

            #base encoder
            encoder11 = self.convblock1(input)
            encoder12 = self.mp1(encoder11)
            encoder21 = self.convblock2(encoder12)
            encoder22 = self.mp2(encoder21)
            encoder31 = self.convblock3(encoder22)
            encoder32 = self.mp3(encoder31)
            encoder41 = self.convblock4(encoder32)
            encoder42 = self.mp4(encoder41)
            encoder5 = self.convblock5(encoder42)

            #vessel extraction decoder
            bridge1 = self.bridgeblock1(encoder5)
            skip11 = torch.cat([bridge1, encoder41], 1)
            decoder11 = self.deconvblock11(skip11)
            skip12 = torch.cat([decoder11, encoder31], 1)
            decoder12 = self.deconvblock12(skip12)
            skip13 = torch.cat([decoder12, encoder21], 1)
            decoder13 = self.deconvblock13(skip13)
            skip14 = torch.cat([decoder13, encoder11], 1)
            output1 = self.outputblock1(skip14)

            # centerness decoder
            bridge2 = self.bridgeblock2(encoder5)
            skip21 = torch.cat([bridge2, encoder41], 1)
            decoder21 = self.deconvblock21(skip21)
            skip22 = torch.cat([decoder21, encoder31], 1)
            decoder22 = self.deconvblock22(skip22)
            skip23 = torch.cat([decoder22, encoder21], 1)
            decoder23 = self.deconvblock23(skip23)
            skip24 = torch.cat([decoder23, encoder11], 1)
            output2 = self.outputblock2(skip24)

            # topological distance decoder
            bridge3 = self.bridgeblock3(encoder5)
            skip31 = torch.cat([bridge3, encoder41], 1)
            decoder31 = self.deconvblock31(skip31)
            skip32 = torch.cat([decoder31, encoder31], 1)
            decoder32 = self.deconvblock32(skip32)
            skip33 = torch.cat([decoder32, encoder21], 1)
            decoder33 = self.deconvblock33(skip33)
            skip34 = torch.cat([decoder33, encoder11], 1)
            output3 = self.outputblock3(skip34)

            return output1, output2, output3


class net_loc(nn.Module):
    def __init__(self, ngpu, ndf = 8, use_bias = True, in_c=1):
        super(net_loc, self).__init__()
        self.ngpu = ngpu

        #base encoder
        self.convblock1 = nn.Sequential(
            nn.Conv3d(in_c, ndf, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace= True),
        )
        self.mp1 = nn.MaxPool3d(2, stride=2)
        self.convblock2 = nn.Sequential(
            nn.Conv3d(ndf, ndf*2, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf*2),
            nn.ReLU(inplace= True),
        )
        self.mp2 = nn.MaxPool3d(2, stride=2)
        self.convblock3 = nn.Sequential(
            nn.Conv3d(ndf*2, ndf*4, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf*4),
            nn.ReLU(inplace= True),
        )
        self.mp3 = nn.MaxPool3d(2, stride=2)
        self.convblock4 = nn.Sequential(
            nn.Conv3d(ndf*4, ndf*8, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf*8),
            nn.ReLU(inplace= True),
        )
        self.mp4 = nn.MaxPool3d(2, stride=2)
        self.convblock5 = nn.Sequential(
            nn.Conv3d(ndf*8, ndf*16, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf*16),
            nn.ReLU(inplace= True),
        )

        #localization decoder
        self.bridgeblock1 = nn.Sequential(
            nn.Conv3d(ndf * 16, ndf*8, 1, 1, 0, bias=use_bias),
            nn.ConvTranspose3d(ndf*8, ndf * 8, kernel_size=4, padding=1,stride=2, bias=use_bias),
        )
        self.deconvblock11 = nn.Sequential(
            nn.Conv3d(ndf * 8 + ndf * 8, ndf * 8, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 8, ndf * 8, kernel_size=4, padding=1, stride=2,bias=use_bias),
        )
        self.deconvblock12 = nn.Sequential(
            nn.Conv3d(ndf * 8 + ndf * 4, ndf * 4, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 4, ndf * 4, kernel_size=4, padding=1, stride=2,bias=use_bias),
        )
        self.deconvblock13 = nn.Sequential(
            nn.Conv3d(ndf * 4 + ndf * 2, ndf * 2, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(ndf * 2, ndf * 2, kernel_size=4, padding=1, stride=2,bias=use_bias),
        )
        self.outputblock1 = nn.Sequential(
            nn.Conv3d(ndf * 2 + ndf, ndf, 3, 1, 1, bias=use_bias),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, 3, 1, 1, 0, bias=use_bias),
            nn.Softmax(dim=1),
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

            #base encoder
            encoder11 = self.convblock1(input)
            encoder12 = self.mp1(encoder11)
            encoder21 = self.convblock2(encoder12)
            encoder22 = self.mp2(encoder21)
            encoder31 = self.convblock3(encoder22)
            encoder32 = self.mp3(encoder31)
            encoder41 = self.convblock4(encoder32)
            encoder42 = self.mp4(encoder41)
            encoder5 = self.convblock5(encoder42)

            #localization decoder
            bridge1 = self.bridgeblock1(encoder5)
            skip11 = torch.cat([bridge1, encoder41], 1)
            decoder11 = self.deconvblock11(skip11)
            skip12 = torch.cat([decoder11, encoder31], 1)
            decoder12 = self.deconvblock12(skip12)
            skip13 = torch.cat([decoder12, encoder21], 1)
            decoder13 = self.deconvblock13(skip13)
            skip14 = torch.cat([decoder13, encoder11], 1)
            output1 = self.outputblock1(skip14)


            return output1,

