import functools
from .ops import conv_norm_relu, dconv_norm_relu, ResidualBlock, get_norm_layer, init_network
import torch
from torch import nn, einsum
from einops import rearrange

def keep_largest_mask_black_new(image):
    mask = image.clone()
    white = torch.ones((1, image.size()[-1], image.size()[-1]), requires_grad=False,
                       dtype=torch.float).to("cuda:0")
    black = torch.zeros((1, image.size()[-1], image.size()[-1]), requires_grad=False,
                        dtype=torch.float).to("cuda:0")
    for i in range(mask.size()[0]):
        mask[i] = torch.where(mask[i] >= (mask[i].min() + 0.1), torch.where(mask[i] < (mask[i].min() + 0.1), mask[i], white),
                           black)
    return mask.detach()

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False,
                 innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [nn.ReLU(True), upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            #x1 = self.model(x)
            #output = 6 * torch.tanh(x1)*mask + x
            #new
            #output = 6 * torch.tanh(x1) * tgm.image.gaussian_blur(mask, (3, 3), (1, 1)) + x
            #return output
            return self.model(x)##
        else:
            #mask = F.interpolate(mask, size=[x.size()[-2],x.size()[-1]])
            return torch.cat([x, self.model(x)], 1)

class UnetSkipConnectionBlock1(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False,
                 innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock1, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [nn.ReLU(True), upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            x1 = self.model(x)
            output = 2 * x1 *keep_largest_mask_black_new(x) + x
            #output = 6 * x1 + x
            #new
            #output = 6 * torch.tanh(x1) * tgm.image.gaussian_blur(mask, (3, 3), (1, 1)) + x
            return output
            #return self.model(x)##
        else:
            #mask = F.interpolate(mask, size=[x.size()[-2],x.size()[-1]])
            return torch.cat([x, self.model(x)], 1)

class UnetSkipConnectionBlock2(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False,
                 innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock2, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downconv]
            up = [nn.ReLU(True), upconv]
            model = down + [submodule] + up
            self.lin = nn.Linear(inner_nc*128*128, inner_nc, bias=use_bias)
            upconv2 = nn.Conv2d(inner_nc, 1, kernel_size=3, stride=1, padding=1)
            self.up2 =nn.Sequential(nn.ReLU(True), upconv2, nn.Tanh())
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            x1 = self.model(x)
            x1f = torch.flatten(x1, 1)
            x2 = self.lin(x1f)
            x3 = x1 * x2[:, :, None, None]
            x4 = self.up2(x3)
            output = 2 * x4 *keep_largest_mask_black_new(x) + x
            return output
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)
        self.unet_model = unet_block

    def forward(self, input):
        return self.unet_model(input)

class UnetGenerator1(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator1, self).__init__()

        unet_block = UnetSkipConnectionBlock1(ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock1(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock1(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock1(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock1(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock1(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)
        self.unet_model = unet_block

    def forward(self, input):
        return self.unet_model(input)

class UnetGenerator2(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator2, self).__init__()

        unet_block = UnetSkipConnectionBlock2(ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock2(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock2(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)
        self.unet_model = unet_block

    def forward(self, input):
        return self.unet_model(input)

class ResnetGenerator(nn.Module):  # todo change here number of channel
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, num_blocks=6):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        res_model = [nn.ReflectionPad2d(3),
                     conv_norm_relu(input_nc, ngf * 1, 7, norm_layer=norm_layer, bias=use_bias),
                     conv_norm_relu(ngf * 1, ngf * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias),
                     conv_norm_relu(ngf * 2, ngf * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)]

        for i in range(num_blocks):
            res_model += [ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)]

        res_model += [dconv_norm_relu(ngf * 4, ngf * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      dconv_norm_relu(ngf * 2, ngf * 1, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      nn.ReflectionPad2d(3),
                      nn.Conv2d(ngf, output_nc, 7),
                      nn.Tanh()]
        self.res_model = nn.Sequential(*res_model)
        self.tangent = nn.Tanh()

    def forward(self, x):
        x1=self.res_model(x)
        output = self.tangent(2*x1 + x)
        return output, keep_largest_mask_black_new(x)
        #output=self.tangent(2*x1*keep_largest_mask_black_new(x)+x)
        #return output, keep_largest_mask_black_new(x)
        #return self.res_model(x)#, keep_largest_mask_black_new(x)
        
        
class Encoder(nn.Module):
    def __init__(self, input_nc, ngf=16, norm_layer=nn.BatchNorm2d, n_downsampling=3):
        super(Encoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        self.down_sampling = nn.Sequential(*model)

    def forward(self, input):
        input = self.down_sampling(input)
        return input


class Decoder(nn.Module):
    def __init__(self, output_nc, ngf=16, norm_layer=nn.BatchNorm2d, n_downsampling=3):
        super(Decoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        self.up_sampling = nn.Sequential(*model)

    def forward(self, input):
        input = self.up_sampling(input)
        return input


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_q = DepthWiseConv2d(dim, inner_dim, 3, padding=padding, stride=1, bias=False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding=padding, stride=kv_proj_stride, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, y=y)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride, heads=heads,
                                       dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ConvTransformer(nn.Module):
    def __init__(self, input_nc, n_downsampling, depth, heads, proj_kernel=3,
                 mlp_mult=4, dropout=0., ngf=16):

        super().__init__()
        dim = (2 ** n_downsampling) * ngf
        self.conv_encoder = Encoder(input_nc=input_nc, ngf=ngf, n_downsampling=n_downsampling)
        self.conv_decoder = Decoder(output_nc=input_nc, ngf=ngf, n_downsampling=n_downsampling)

        self.transformer = Transformer(dim=dim, proj_kernel=proj_kernel, kv_proj_stride=2, depth=depth, heads=heads,
                                       mlp_mult=mlp_mult, dropout=dropout)

    def forward(self, img):
        x = self.conv_encoder(img)
        x = self.transformer(x)
        x = self.conv_decoder(x)
        return x



def define_Gen(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, gpu_ids=[0]):
    gen_net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        gen_net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                  num_blocks=9)
    elif netG == 'resnet_6blocks':
        gen_net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                  num_blocks=6)
    elif netG == 'unet_128':
        gen_net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        gen_net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_128_attentionmap':
        gen_net = UnetGenerator2(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_128_mask':
        gen_net = UnetGenerator1(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'transgan':
        gen_net = ConvTransformer(input_nc, 3, 3, 6, ngf=ngf, dropout=use_dropout)   
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_network(gen_net, gpu_ids)

