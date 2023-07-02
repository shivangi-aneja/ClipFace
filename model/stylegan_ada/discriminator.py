# Code from https://github.com/nihalsid/stylegan2-ada-lightning

import torch
import numpy as np
from model.stylegan_ada import SmoothDownsample, EqualizedConv2d, FullyConnectedLayer
from model.stylegan_ada.generator import MappingNetwork


class Discriminator(torch.nn.Module):

    def __init__(self, img_resolution, img_channels, w_num_layers, exp_dim=0, exp_map_dim=512, channel_base=16384, channel_max=512):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.exp_map_dim = 0
        self.exp_dim = exp_dim
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        self.num_ws = 2 * (len(self.block_resolutions) + 1)

        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        module_list = [EqualizedConv2d(img_channels, channels_dict[img_resolution], kernel_size=1, activation='lrelu')]
        for res in self.block_resolutions:
            in_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            module_list.append(DiscriminatorBlock(in_channels, out_channels))
        self.net = torch.nn.Sequential(*module_list)

        if self.exp_dim > 0:
            self.exp_map_dim = exp_map_dim
            self.mapping = MappingNetwork(z_dim=0, w_dim=self.exp_map_dim, exp_dim=exp_dim, num_ws=None, num_layers=w_num_layers, w_avg_beta=None)

        self.b4 = DiscriminatorEpilogue(channels_dict[4], exp_map_dim=self.exp_map_dim, resolution=4)

    def forward(self, x, exp_code=None):
        x = self.net(x)

        # Conditioning
        exp_map_feat = None
        if self.exp_map_dim > 0:
            exp_map_feat = self.mapping(None, exp_code)

        x = self.b4(x, exp_map_feat)
        return x


class DiscriminatorBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = 0
        downsampler = SmoothDownsample()
        self.conv0 = EqualizedConv2d(in_channels, in_channels, kernel_size=3, activation=activation)
        self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, activation=activation, resample=downsampler)
        self.skip = EqualizedConv2d(in_channels, out_channels, kernel_size=1, bias=False, resample=downsampler)

    def forward(self, x):
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        x = y.add_(x)
        return x


class DiscriminatorEpilogue(torch.nn.Module):

    def __init__(self, in_channels, resolution, exp_map_dim=0, mbstd_group_size=4, mbstd_num_channels=1, activation='lrelu'):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.exp_map_dim = exp_map_dim

        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = EqualizedConv2d(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if exp_map_dim == 0 else exp_map_dim)

    def forward(self, x, exp_map_feat=None):
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.exp_map_dim > 0:
            x = (x * exp_map_feat).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.exp_map_dim))
        return x


class MinibatchStdLayer(torch.nn.Module):

    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x


if __name__ == '__main__':
    from util.misc import print_model_parameter_count, print_module_summary

    model = Discriminator(img_resolution=64, img_channels=3)
    print_module_summary(model, (torch.randn((16, 3, 64, 64)), ))
    print_model_parameter_count(model)
