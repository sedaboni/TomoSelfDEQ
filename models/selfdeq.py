import torch
from torch.nn.functional import pad
from torch.nn.utils import spectral_norm

activation_fn = {
    'relu': lambda: torch.nn.ReLU(inplace=True),
    'lrelu': lambda: torch.nn.LeakyReLU(inplace=True),
    'prelu': lambda: torch.nn.PReLU(),
    'selu': lambda: torch.nn.SELU(),
}

class UNet(torch.nn.Module):
    def __init__(self, dimension, i_nc=1, o_nc=1, f_root=32, conv_times=2, is_bn=False, activation='relu',
                 is_residual=False, up_down_times=3, is_spe_norm=True, padding=(0, 0)):

        self.is_residual = is_residual
        self.up_down_time = up_down_times
        self.dimension = dimension
        self.padding = padding

        super().__init__()

        self.relu = torch.nn.ReLU()

        if dimension == 2:
            self.down_sample = torch.nn.MaxPool2d((2, 2))
        elif dimension == 3:
            self.down_sample = torch.nn.MaxPool3d((1, 2, 2))
        else:
            raise ValueError()

        self.conv_in = ConvBnActivation(
            in_channels=i_nc,
            out_channels=f_root,
            is_bn=is_bn,
            activation=activation,
            dimension=dimension,
            is_spe_norm=is_spe_norm
        )

        self.conv_out = ConvBnActivation(
            in_channels=f_root,
            out_channels=o_nc,
            kernel_size=1,
            dimension=dimension,
            times=1,
            is_bn=False,
            activation=None,
            is_spe_norm=is_spe_norm
        )

        self.bottom = ConvBnActivation(
            in_channels=f_root * (2 ** (up_down_times - 1)),
            out_channels=f_root * (2 ** up_down_times),
            times=conv_times, is_bn=is_bn, activation=activation, dimension=dimension,
            is_spe_norm=is_spe_norm
        )

        self.down_list = torch.nn.ModuleList([
                                           ConvBnActivation(
                                               in_channels=f_root * 1,
                                               out_channels=f_root * 1,
                                               times=conv_times, is_bn=is_bn, activation=activation,
                                               dimension=dimension,
                                               is_spe_norm=is_spe_norm
                                           )
                                       ] + [
                                           ConvBnActivation(
                                               in_channels=f_root * (2 ** i),
                                               out_channels=f_root * (2 ** (i + 1)),
                                               times=conv_times, is_bn=is_bn, activation=activation,
                                               dimension=dimension,
                                               is_spe_norm=is_spe_norm
                                            )
                                           for i in range(up_down_times - 1)
                                       ])

        self.up_conv_list = torch.nn.ModuleList([
            ConvBnActivation(
                in_channels=f_root * (2 ** (up_down_times - i)),
                out_channels=f_root * (2 ** (up_down_times - i - 1)),
                times=conv_times, is_bn=is_bn, activation=activation, dimension=dimension,
                is_spe_norm=is_spe_norm
            )
            for i in range(up_down_times)
        ])

        self.up_conv_tran_list = torch.nn.ModuleList([
            ConvtranBnActivation(
                in_channels=f_root * (2 ** (up_down_times - i)),
                out_channels=f_root * (2 ** (up_down_times - i - 1)),
                is_bn=is_bn, activation=activation, dimension=dimension,
                is_spe_norm=is_spe_norm
            )
            for i in range(up_down_times)
        ])

    def forward(self, x):

        input_ = x

        x = pad(x, [0, self.padding[0], 0, self.padding[1]])

        x = self.conv_in(x)

        skip_layers = []
        for i in range(self.up_down_time):
            x = self.down_list[i](x)

            skip_layers.append(x)
            x = self.down_sample(x)

        x = self.bottom(x)

        for i in range(self.up_down_time):
            x = self.up_conv_tran_list[i](x)
            x = torch.cat([x, skip_layers[self.up_down_time - i - 1]], 1)
            x = self.up_conv_list[i](x)

        x = self.conv_out(x)

        if self.padding[0] > 0:
            x = x[..., :-self.padding[0]]
        if self.padding[1] > 0:
            x = x[..., :-self.padding[1], :]

        # x = x[..., :-self.padding[1], :-self.padding[0]]

        ret = input_ - x if self.is_residual else x

        return ret
        # return torch.pow(ret,2)
    
class ConvBnActivation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, times=1, is_bn=False, activation='relu', kernel_size=3, is_spe_norm=False):
        super().__init__()

        if dimension == 3:
            conv_fn = lambda in_c: torch.nn.Conv3d(in_channels=in_c,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size // 2
                                                   )
            bn_fn = lambda: torch.nn.BatchNorm3d(out_channels)

        elif dimension == 2:
            conv_fn = lambda in_c: torch.nn.Conv2d(in_channels=in_c,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size // 2
                                                   )
            bn_fn = lambda: torch.nn.BatchNorm2d(out_channels)
        else:
            raise ValueError()

        layers = []
        for i in range(times):
            if i == 0:
                layers.append(spectral_norm(conv_fn(in_channels)) if is_spe_norm else conv_fn(in_channels))
            else:
                layers.append(spectral_norm(conv_fn(out_channels)) if is_spe_norm else conv_fn(out_channels))

            if is_bn:
                layers.append(bn_fn())

            if activation is not None:
                layers.append(activation_fn[activation]())

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())
    

    
class ConvtranBnActivation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, is_bn=False, activation='relu', kernel_size=3, is_spe_norm=False):
        self.is_bn = is_bn
        super().__init__()
        if dimension == 3:
            conv_fn = lambda: torch.nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 2, 2),
                padding=kernel_size // 2,
                output_padding=(0, 1, 1)
            )
            bn_fn = lambda: torch.nn.BatchNorm3d(out_channels)

        elif dimension == 2:
            conv_fn = lambda: torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1
            )
            bn_fn = lambda: torch.nn.BatchNorm2d(out_channels)
        else:
            raise ValueError()

        self.net1 = spectral_norm(conv_fn()) if is_spe_norm else conv_fn()
        if self.is_bn:
            self.net2 = bn_fn()
        self.net3 = activation_fn[activation]()

    def forward(self, x):
        ret = self.net1(x)
        if self.is_bn:
            ret = self.net2(ret)

        ret = self.net3(ret)

        return ret
