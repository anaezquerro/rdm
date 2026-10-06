from torch import nn
import torch 

from .block import UNetBlock


class Bottleneck(nn.Module):
    def __init__(self, input_dim: int, cross_attn: bool, shared_attn: bool, **_):
        super(Bottleneck, self).__init__()
        self.input_dim = input_dim 
        self.output_dim = input_dim 

        self.blocks = nn.ModuleList([
            UNetBlock(input_dim, input_dim, num_blocks=1, self_attn=True, cross_attn=cross_attn, shared_attn=shared_attn, **_),
            UNetBlock(input_dim, input_dim, num_blocks=1, self_attn=False, cross_attn=False, shared_attn=False, **_)
        ])

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, **_)
        return x
