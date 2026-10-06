from math import ceil
from torch import nn
import torch 

from .sample import Downsample
from .block import UNetBlock

from src.neural.conv import CoordConv
from src.util import Config


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        self_attn: list[bool],
        cross_attn: list[bool],
        shared_attn: list[bool],
        num_blocks: int | list[int],
        down: Config,
        use_coords: bool,
        **_
    ):
        super(Encoder, self).__init__()
        self.input_dim = input_dim 

        if use_coords:
            self.input = CoordConv(input_dim, hidden_dims[0], 3, 1, padding=1)
        else:
            self.input = nn.Conv2d(input_dim, hidden_dims[0], 3, 1, padding=1)
        self.blocks = nn.ModuleList([])
        self.down = nn.ModuleList([])
        self.hidden_dims = [hidden_dims[0]]
        cur_dim = hidden_dims[0]
        self.num_downsamples = 0
        for level, hidden_dim in enumerate(hidden_dims):
            self.blocks.append(
                UNetBlock(
                    cur_dim, hidden_dim, num_blocks[level], 
                    self_attn=self_attn[level], 
                    cross_attn=cross_attn[level], 
                    shared_attn=shared_attn[level],
                    **_
                )
            )
            cur_dim = hidden_dim
            if level != len(hidden_dims) - 1:
                self.down.append(Downsample(input_dim=cur_dim, **down))
            else:
                self.down.append(nn.Identity())
            self.hidden_dims.append(cur_dim)
        self.output_dim = cur_dim

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim}, hidden_dims={self.hidden_dims})'

    def output_size(self, resolution: tuple[int, int]) -> tuple[int, int, int]:
        """Returns the output size of the encoder (d, h, w)."""
        res = []
        for s in resolution:
            for _ in range(self.num_downsamples):
                s = ceil(s / 2)
            res.append(s)
        return self.output_dim, *res

    def forward(self, x: torch.Tensor, **_) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
            x (torch.Tensor ~ [batch_size, input_dim, h, w]): Input feature map tensor.
            
        Returns:
            Final feature map and residual connections.
        """
        h = self.input(x)
        hs = []
        for block, down in zip(self.blocks, self.down):
            h = block(h, **_)
            hs.append(h)
            h = down(h)
        return h, hs

