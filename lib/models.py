import logging
from typing import List, Tuple

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
    def __init__(
        self, Ks: int, Kt: int, hist_window: int, pred_window: int, encoder_blocks: List[int],
        decoder_blocks: List[Tuple[int]]
    ) -> None:
        super(STGCN_VAE, self).__init__()
        self.encoder = nn.ModuleList(
            [layers.SpatioTemporalConv(Ks, Kt, channels, 0.1, act='relu') for channels in encoder_blocks]
        )
        _, _, out_channels = encoder_blocks[-1]
        self.gconv_mu = GCNConv(out_channels, 1, cached=True, improved=True)
        self.gconv_var = GCNConv(out_channels, 1, cached=True, improved=True)

        _dec_layers = [
            layers.ResidualGConv(decoder_blocks[i] + hist_window, decoder_blocks[i + 1], 'relu')
            for i in range(len(decoder_blocks) - 1)
        ]
        self.decoder = nn.ModuleList(
            [*_dec_layers, layers.ResidualGConv(decoder_blocks[-1] + hist_window, pred_window, 'relu')]
        )

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu if logvar is None else (mu + torch.rand_like(logvar) * logvar.mul(0.5).exp())

    def encode(self, x: torch.Tensor, y: torch.Tensor, edge_idx: torch.Tensor,
               edge_wt: torch.Tensor) -> Tuple[torch.Tensor]:
        theta = torch.cat((x, y), axis=-1)
        for enc in self.encoder:
            theta = enc(theta, edge_idx, edge_wt)
        mu, logvar = self.gconv_mu(theta, edge_idx, edge_wt), self.gconv_var(theta, edge_idx, edge_wt)
        logger.debug("Distribution: mu=%s; logvar=%s", mu.shape, logvar.shape)
        return mu, logvar

    def decode(
        self,
        mu: torch.Tensor,
        std: torch.Tensor,
        x: torch.Tensor,
        edge_idx: torch.Tensor,
        edge_wt: torch.Tensor,
    ) -> torch.Tensor:
        z = self.reparametrize(mu, std)
        logger.debug("Latent variable, z: %s", z.shape)
        y_hat = z
        for block in self.decoder:
            recon = torch.cat((y_hat, x), axis=-1)
            y_hat = block(recon, edge_idx, edge_wt)
        return y_hat

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_idx: torch.Tensor,
        edge_wt: torch.Tensor,
        sample: bool = True
    ) -> torch.Tensor:
        mu, logvar = self.encode(x, y, edge_idx, edge_wt)
        y_hat = self.decode(mu, logvar if sample else None, x, edge_idx, edge_wt)
        return y_hat