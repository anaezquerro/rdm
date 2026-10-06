
from typing import Callable
from torch import nn 
import torch 

from src.util import Config 
from src.neural.sequential import Identity
from .attn import MultiHeadSpatialAttention, MultiHeadSharedCrossAttention, MultiHeadSpatialCrossAttention
from .residual import ResidualBlock

class UNetBlock(nn.Module):
    """Implementation of the U-Net block, integrating:
    - Residual connections.
    - Self-attention.
    - Cross-attention with an specific feature.
    - Time embedding condtioning.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_blocks: int,
        residual: Config,
        attention: Config,
        time_dim: int = None,
        self_attn: bool = False,
        cross_attn: bool = False,
        shared_attn: bool = False,
        **_
    ):
        super().__init__()
        assert not (shared_attn and cross_attn), "Cannot use both shared and cross attention in the same block."
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.time_dim = time_dim
    
        self.blocks = nn.ModuleList([])
        dim = input_dim
        for _ in range(num_blocks):
            self.blocks.append(ResidualBlock(dim, output_dim, time_dim=time_dim, **residual))
            dim = output_dim

        if self_attn:
            self.attn = MultiHeadSpatialAttention(output_dim, **attention)
        else:
            self.attn = Identity()
        if cross_attn:
            self.cross = MultiHeadSpatialCrossAttention(output_dim, **attention)
        elif shared_attn:
            self.cross = MultiHeadSharedCrossAttention(output_dim, **attention)
        else:
            self.cross = Identity()

    def forward(
        self, 
        x: torch.Tensor, 
        temb: Callable = None, 
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        pos: Callable = None,
        feat: torch.Tensor = None,
        **_
    ) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor ~ [batch_size, input_dim, h, w]): Input feature map tensor.
            temb (Callable): Dynamic time embeddings.
            key (torch.Tensor ~ [batch_size, num_feats, dim]): Key features for shared cross attention.
            value (torch.Tensor ~ [batch_size, num_feats, dim]): Value features for shared cross attention.
            pos (torch.Tensor ~ [*grid, dim]): Positional features.
            feat (torch.Tensor ~ [batch_size, num_feats, dim]): Conditional features.

        Returns:
            torch.Tensor: Output feature map tensor.
        """
        for block in self.blocks:
            x = block(x, temb)
        x = self.attn(x)
        x = self.cross(x, key=key, value=value, pos=pos, feat=feat)
        return x 

