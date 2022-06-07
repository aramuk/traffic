from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv

from . import logs

ACTIVATION_FNS = {
    'glu': F.glu,
    'relu': F.relu,
    'linear': None,
    'sigmoid': F.sigmoid,
}

logger = logs.get_logger()

class Align(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Align, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding='valid')

    def forward(self, x: torch.Tensor):
        if self.in_channels > self.out_channels:
            out = torch.permute(x, (0, 3, 1, 2))
            out = self.conv(out)
            return torch.permute(out, (0, 2 , 3, 1))
        elif self.in_channels < self.out_channels:
            batch_size, T, N, _ = x.shape
            return torch.cat((x, torch.zeros(batch_size, T, N, self.out_channels - self.in_channels)), dim=3)
        return x


class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, act: str) -> None:
        super(TemporalConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.act = act
        self.align = Align(in_channels, 2 * out_channels if act == 'glu' else out_channels)
        self.gconv = ChebConv(in_channels, 2 * out_channels if act == 'glu' else out_channels, K=kernel_size)

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor, edge_wt: torch.Tensor) -> torch.Tensor:
        logger.debug("Into temporal conv: %s", x.shape)
        x_in = self.align(x)
        x_out = self.gconv(x, edge_idx, edge_wt)
        if fn := ACTIVATION_FNS.get(self.act):
            if self.act == 'glu':
                x_p = x_out[:, :self.out_channels, :, :]
                x_q = x_out[:, -self.out_channels:, :, :]
                x_out = fn(torch.cat((x_p + x_in, x_q), dim=3))
            else:
                x_out = fn(x_out + x_in)
        logger.debug("Out of temporal conv: %s", x_out.shape)
        return x_out


class SpatialConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super(SpatialConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.align = Align(in_channels, out_channels)
        self.gconv = ChebConv(in_channels, out_channels, K=kernel_size)

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor, edge_wt: torch.Tensor) -> torch.Tensor:
        _, T, N, _ = x.shape
        logger.debug("Into spatial conv: %s", x.shape)
        x_out = self.align(x)
        x_out = self.gconv(x_out, edge_idx, edge_wt)
        x_out = torch.reshape(x_out, (-1, T, N, self.out_channels))
        x_out = F.relu(x_out)
        logger.debug("Out of spatial conv: %s", x_out.shape)
        return x_out


class SpatioTemporalConv(nn.Module):
    def __init__(
        self, spatial_kernel: int, temporal_kernel: int, channels: Tuple[int], dropout_p: float, act: str
    ) -> None:
        super(SpatioTemporalConv, self).__init__()
        spatio_channels, temporal_channels, out_channels = channels
        self.temporal_conv1 = TemporalConv(spatio_channels, temporal_channels, temporal_kernel, act)
        self.spatial_conv = SpatialConv(temporal_channels, temporal_channels, spatial_kernel)
        self.temporal_conv2 = TemporalConv(temporal_channels, out_channels, temporal_kernel, act)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor, edge_wt: torch.Tensor) -> torch.Tensor:
        x_out = x
        logger.debug("Into ST block: %s", x_out.shape)
        x_out = self.temporal_conv1(x_out, edge_idx, edge_wt)
        x_out = self.spatial_conv(x_out, edge_idx, edge_wt)
        x_out = self.temporal_conv2(x_out, edge_idx, edge_wt)
        x_out = self.norm(x_out)
        x_out = self.dropout(x_out)
        logger.debug("Out of ST block: %s", x_out.shape)
        return x_out


class Classifier(nn.Module):
    def __init__(self, channels: int, kernel_size: int, act: str) -> None:
        super(Classifier, self).__init__()
        self.temporal_conv1 = TemporalConv(channels, channels, kernel_size, act)
        self.norm = nn.LayerNorm(channels)
        self.temporal_conv2 = TemporalConv(channels, channels, 1, 'sigmoid')
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = x
        x_out = self.temporal_conv1(x_out)
        x_out = self.norm(x_out)
        x_out = self.temporal_conv2(x_out)
        x_out = self.conv(x_out)
        return x_out


class ResidualGConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str) -> None:
        super(ResidualGConv, self).__init__()
        self.gconv = GCNConv(in_channels, out_channels, improved=True)
        self.align = Align(in_channels, out_channels)
        self.act = act

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor, edge_wt: torch.Tensor) -> torch.Tensor:
        logger.debug("Into residual conv: %s", x.shape)
        x_in = self.align(x)
        x_out = self.gconv(x, edge_idx, edge_wt)
        if fn := ACTIVATION_FNS.get(self.act):
            x_out = fn(x_out)
        x_out = x_in + x_out
        logger.debug("Out of residual conv: %s", x_out.shape)
        return x_out
