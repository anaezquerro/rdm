from __future__ import annotations
from torch import nn 
from math import prod 
import torch 

from .encoder import Encoder
from .decoder import Decoder 
from .bottleneck import Bottleneck


class UNet(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        output_dim: int,
        hidden_dims: list[int],
        num_blocks: int | list[int],
        bottleneck: bool = True,
        self_attn: bool | list[bool] = False,
        cross_attn: list[bool] = False,
        shared_attn: list[bool] = False,
        **conf
    ):
        """Implementation of the standard U-Net with self-attention and cross-attention layers.""" 
        super(UNet, self).__init__()
        self.input_size = self.input_dim, *self.resolution = input_size
        self.output_dim = output_dim
        self_attn = [self_attn]*len(hidden_dims) if isinstance(self_attn, bool) else self_attn
        cross_attn = [cross_attn]*(len(hidden_dims)+1) if isinstance(cross_attn, bool) else cross_attn
        shared_attn = [shared_attn]*(len(hidden_dims)+1) if isinstance(shared_attn, bool) else shared_attn
        num_blocks = [num_blocks]*len(hidden_dims) if isinstance(num_blocks, int) else num_blocks

        self.encoder = Encoder(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            self_attn=self_attn, 
            cross_attn=cross_attn,
            shared_attn=shared_attn,
            num_blocks=num_blocks,
            **conf,
        )
        if bottleneck:
            self.bottleneck = Bottleneck(
                input_dim=self.encoder.output_dim, 
                cross_attn=cross_attn[-1],
                shared_attn=shared_attn[-1],
                **conf 
            )
        else:
            self.bottleneck = lambda x, *y: x 

        self.decoder = Decoder(
            input_dims=hidden_dims[::-1], 
            output_dim=output_dim,
            self_attn=self_attn[::-1],
            cross_attn=cross_attn[:-1][::-1],
            shared_attn=shared_attn[:-1][::-1],
            num_blocks=num_blocks[::-1],
            **conf,
        )
        
    @property
    def num_feats(self) -> int:
        return prod(self.encoder.output_size(self.resolution)[1:])
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor ~ [batch_size, input_dim, height, width]): Input feature map.
            
        Returns: 
            torch.Tensor ~ [batch_size, output_dim, height, width]: Output feature map.
        """
        h, hs = self.encoder(x)
        h = self.bottleneck(h)
        out = self.decoder(h, hs)
        return out 
    
