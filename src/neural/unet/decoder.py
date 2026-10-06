from copy import deepcopy
from torch import nn
import torch

from .sample import Upsample
from .block import UNetBlock
from src.util import Config
from src.neural import Norm, Activation
from src.neural.sequential import Identity


class Decoder(nn.Module):
    def __init__(
        self,
        input_dims: list[int],
        output_dim: int,
        self_attn: list[bool],
        cross_attn: list[bool],
        shared_attn: list[bool],
        num_blocks:  list[int],
        up: Config,
        out: Config,
        **_
    ):
        super(Decoder, self).__init__()
        self.input_dims = input_dims 
        self.output_dim = output_dim
        self.output_dim = output_dim 

        self.blocks = nn.ModuleList([])
        self.up = nn.ModuleList([])
        for i, dim in enumerate(input_dims):
            self.blocks.append(
                UNetBlock(
                    dim*2, 
                    input_dims[min(i+1, len(input_dims)-1)],
                    num_blocks[i],
                    self_attn=self_attn[i], 
                    cross_attn=cross_attn[i],
                    shared_attn=shared_attn[i],
                    **_
                )
            )
            if i != len(input_dims) - 1:
                self.up.append(Upsample(input_dim=input_dims[min(i+1, len(input_dims)-1)], **up))
            else:
                self.up.append(Identity())

        self.out = nn.Sequential(
            Norm(out.norm, dim),
            Activation(out.activation),
            nn.Conv2d(dim, output_dim, 3, 1, padding=1)
        )
        self.init_weights()
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(input_dims={self.input_dims}'

    def init_weights(self):
        for n, m in self.named_modules():
            if (isinstance(m, nn.Conv2d) and ('conv_2' in n or 'out' in n)) \
                or ( isinstance(m, nn.Conv1d) and 'proj' in n):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        h: torch.Tensor,
        hs: list[torch.Tensor],
        **_
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            h (torch.Tensor ~ [batch_size, input_dim, h, w]): Input feature map.
            hs (list[torch.Tensor]): Residual connections.
            temb (Callable -> torch.Tensor ~ [batch_size, heigh, width, cond_dim]): Time embeddings.
            feat (torch.Tensor ~ [batch_sie, num_feats, cond_dim]): Hidden features for cross attention.

        Returns:
            torch.Tensor ~ [batch_size, output_dim, height, width]
        """
        for block, up in zip(self.blocks, self.up):
            h = block(torch.cat((h, hs.pop()), dim=1), **_)
            h = up(h, shape=hs[-1].shape[-2:] if hs else None)
        out = self.out(h)
        return out


