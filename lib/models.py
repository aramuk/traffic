import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from . import layers

logger = logging.getLogger("traffic")

class STGCN(nn.Module):
    def __init__(self, Ks, Kt, hist_size, blocks) -> None:
        super(STGCN, self).__init__()
        Ko = hist_size
        self.encoder = nn.Sequential(
            *(layers.SpatioTemporalConv(Ks, Kt, channels, 0.1, act='glu') for channels in blocks[:-1])
        )
        Ko -= len(blocks) * 2 * (Ks - 1)
        self.classifier = layers.Classifier(blocks[-1], Ko, act='glu')

    def forward(self, x, edge_idx, edge_wt):
        features = self.classifier(x, edge_idx, edge_wt)
        y = self.classifier(features)
        return y


class STGCN_VAE(nn.Module):
    def __init__(self, Ks: int, Kt: int, blocks, latent_dim: int) -> None:
        super(STGCN_VAE, self).__init__()
        # Ko = hist_size
        self.latent_dim = latent_dim
        self.encoder = nn.ModuleList(
            [layers.SpatioTemporalConv(Ks, Kt, channels, 0.1, act='relu') for channels in blocks]
        )
        _, _, out_channels = blocks[-1]
        self.conv_mu = GCNConv(out_channels, 1, cached=True, improved=True)
        self.conv_var = GCNConv(out_channels, 1, cached=True, improved=True)

        # Ko -= len(blocks) * 2 * (Ks - 1)
        self.decoder = nn.ModuleList(
            [
                layers.ResidualGConv(1, 16, 'relu'),
                layers.ResidualGConv(16, 32, 'relu'),
                layers.ResidualGConv(32, 1, 'relu')
            ]
        )

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu if logvar is None else (mu + torch.rand_like(logvar) * logvar.mul(0.5).exp())

    def encode(self, x: torch.Tensor, y: torch.Tensor, edge_idx: torch.Tensor,
               edge_wt: torch.Tensor) -> Tuple[torch.Tensor]:
        theta = torch.cat((x, y), axis=-1)
        for enc in self.encoder:
            theta = enc(theta, edge_idx, edge_wt)
        mu, logvar = self.conv_mu(theta, edge_idx, edge_wt), self.conv_var(theta, edge_idx, edge_wt)
        logger.debug("Distribution: mu=%s; logvar=%s", mu.shape, logvar.shape)
        return mu, logvar

    def decode(
        self,
        mu: torch.Tensor,
        std: torch.Tensor = None,
        edge_idx: torch.Tensor = None,
        edge_wt: torch.Tensor = None,
    ) -> torch.Tensor:
        z = self.reparametrize(mu, std)
        logger.debug("Latent variable, z: %s", z.shape)
        y_hat = z
        for dec in self.decoder:
            y_hat = dec(y_hat, edge_idx, edge_wt)
        return y_hat

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_idx: torch.Tensor,
        edge_wt: torch.Tensor,
        sample=True
    ) -> torch.Tensor:
        mu, logvar = self.encode(x, y, edge_idx, edge_wt)
        y_hat = self.decode(mu, logvar if sample else None, edge_idx, edge_wt)
        return y_hat