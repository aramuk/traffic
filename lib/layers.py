from turtle import forward
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

ACTIVATION_FNS = {
    'glu': F.glu,
    'relu': F.relu,
    'linear': None,
    'sigmoid': F.sigmoid,
}


class Align(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Align, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        if self.in_channels > self.out_channels:
            return self.conv(x)
        elif self.in_channels < self.out_channels:
            batch_size, T, N, _ = x.shape
            return torch.concat(x, torch.zeros(batch_size, T, N, self.out_channels - self.in_channels), axis=3)
        return x


class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, act: str) -> None:
        super(TemporalConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.act = act
        self.align = Align(in_channels, out_channels)
        self.gconv = ChebConv(in_channels, 2 * out_channels if act == 'glu' else out_channels, K=kernel_size)

    def forward(self, x) -> torch.Tensor:
        x_in = x
        x_out = self.gconv(x_in)
        if fn := ACTIVATION_FNS.get(self.act):
            if self.act == 'glu':
                x_p = x_out[:, :self.out_channels, :, :]
                x_q = x_out[:, -self.out_channels:, :, :]
                x_out = fn(torch.stack(x_p + x_in, x_q))
            else:
                x_out = fn(x_out + x_in)
        return x_out


class SpatioConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super(SpatioConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.align = Align(in_channels, out_channels)
        self.gconv = ChebConv(in_channels, out_channels, K=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, N, _ = x.shape
        x_out = x
        x_out = self.align(x_out)
        x_out = self.gconv(x_out)
        x_out = torch.reshape(x_out, (-1, T, N, self.out_channels))
        x_out = F.relu(x_out)
        return x_out


class SpatioTemporalConv(nn.Module):
    def __init__(
        self, spatio_kernel: int, temporal_kernel: int, channels: Tuple[int], dropout_p: float, act: str
    ) -> None:
        super(SpatioTemporalConv, self).__init__()
        spatio_channels, temporal_channels, out_channels = channels
        self.spatio_conv = SpatioConv(spatio_channels, temporal_channels, spatio_kernel)
        self.temporal_conv1 = TemporalConv(temporal_channels, temporal_channels, temporal_kernel, act)
        self.temporal_conv2 = TemporalConv(temporal_channels, out_channels, temporal_kernel, act)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = x
        x_out = self.temporal_conv1(x_out)
        x_out = self.spatio_conv(x_out)
        x_out = self.temporal_conv2(x_out)
        x_out = self.norm(x_out)
        x_out = self.dropout(x_out)
        return x_out

class Classifier(nn.Module):
    def __init__(self, channels: int, kernel_size: int, act: str) -> None:
        super(Classifier, self).__init__()
        self.temporal_conv1 = TemporalConv(channels, channels, kernel_size, act)
        self.norm = nn.LayerNorm(channels)
        self.temportal_conv2 = TemporalConv(channels, channels, 1, 'sigmoid')
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = x
        x_out = self.temporal_conv1(x_out)
        x_out = self.norm(x_out)
        x_out = self.temporal_conv2(x_out)
        x_out = self.conv(x_out)
        return x_out