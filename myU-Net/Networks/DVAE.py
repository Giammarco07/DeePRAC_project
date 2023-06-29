import torch.nn as nn
import torch


class DVAE(nn.Module):
    def __init__(self, zdim):
        super().__init__()

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.maxpool = nn.MaxPool3d(kernel_size=2,
                                    stride=2,
                                    padding=0)

        self.enc_l1 = nn.Conv3d(1, 64, 3, stride=1, padding=1, dilation=1)
        self.enc_l2 = nn.Conv3d(64, 64, 3, stride=1, padding=1, dilation=1)
        self.enc_l3 = nn.Conv3d(64, 256, 3, stride=1, padding=1, dilation=1)
        self.enc_l4 = nn.Conv3d(256, 256, 3, stride=1, padding=1, dilation=1)
        self.enc_l51 = nn.Conv3d(256, zdim, 1, stride=1, padding=0, dilation=1)
        self.enc_l52 = nn.Conv3d(256, zdim, 1, stride=1, padding=0, dilation=1)

        self.dec_l1 = nn.ConvTranspose3d(zdim, 256, 4, stride=2, padding=1, output_padding=0)
        self.dec_l2 = nn.ConvTranspose3d(256, 256, 4, stride=2, padding=1, output_padding=0)
        self.dec_l3 = nn.ConvTranspose3d(256, 64, 4, stride=2, padding=1, output_padding=0)
        self.dec_l4 = nn.Conv3d(64, 64, 3, stride=1, padding=1, dilation=1)
        self.dec_l5 = nn.Conv3d(64, 3, 3, stride=1, padding=1, dilation=1)

    def encode(self, x):
        enc_h1 = self.relu(self.enc_l1(x))
        enc_h2 = self.relu(self.enc_l2(self.maxpool(enc_h1)))
        enc_h3 = self.relu(self.enc_l3(self.maxpool(enc_h2)))
        enc_h4 = self.relu(self.enc_l4(self.maxpool(enc_h3)))
        return self.enc_l51(enc_h4), self.enc_l52(enc_h4)

    def sample(self, mu, logvar, phase):
        if phase == 'training':
            std = logvar.exp().sqrt()
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decode(self, z):
        dec_h1 = self.relu(self.dec_l1(z))
        dec_h2 = self.relu(self.dec_l2(dec_h1))
        dec_h3 = self.relu(self.dec_l3(dec_h2))
        dec_h4 = self.relu(self.dec_l4(dec_h3))
        return self.softmax(self.dec_l5(dec_h4))

    def forward(self, x, phase):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar, phase)
        ref = self.decode(z)
        return ref, mu, logvar, z