
import torch
from torch import nn
from typing import Callable
from src.neural import Norm, Activation


class NormWithEmbedding(nn.Module):

    def __init__(
        self,
        input_dim: int,
        time_dim: int,
        norm: str = 'group',
        activation: str = 'silu',
        use_scale_shift: bool = True
    ):
        """Normalization with embedding layer. If `use_scale_shift == True`,
        embedding results will be chunked and used to re-shift and re-scale
        normalization results. Otherwise, embedding results will directly add to
        input of normalization layer.

        Args:
            input_dim (int): Number of channels of the input feature map.
            cond_dim (int): Size of conditional embedding.
            norm (str): Type of layer normalization.
            act (str): Activation layers.
            use_scale_shift (bool): If True, the output of Embedding layer will be
                split to 'scale' and 'shift' and map the output of normalization
                layer to ``out * (1 + scale) + shift``. Otherwise, the output of
                Embedding layer will be added with the input before normalization
                operation. Defaults to True.
        """
        super(NormWithEmbedding, self).__init__()
        self.input_dim = input_dim
        self.cond_dim = time_dim
        self.use_scale_shift = use_scale_shift
        self.norm = Norm(norm, input_dim)

        emb_output = input_dim * 2 if use_scale_shift else input_dim
        self.embedding_layer = nn.Sequential(
            Activation(activation),
            nn.Linear(time_dim, emb_output)
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(input_dim={self.input_dim}, cond_dim={self.cond_dim}, use_scale_shift={self.use_scale_shift})'

    def forward(self, x: torch.Tensor, temb: Callable):
        """Forward function.

        Args:
            x (torch.Tensor ~ [batch_size, input_dim, h, w]): Input feature map tensor.
            temb (Callable -> torch.Tensor ~ [batch_size, h, w, cond_dim]): Time embeddings.

        Returns:
            torch.Tensor : Output feature map tensor.
        """
        emb = temb(x.shape[-2:])
        emb = self.embedding_layer(emb).permute(0, 3, 1, 2)
        if self.use_scale_shift:
            scale, shift = torch.chunk(emb, 2, dim=1)
            x = self.norm(x)
            x = x * (1 + scale) + shift
        else:
            x = self.norm(x + emb)
        return x


class ResidualBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        norm: str = 'group',
        activation: str = 'silu',
        shortcut_kernel_size: int = 1,
        dropout: float = 0.0,
        time_dim: int = 0,
        use_scale_shift_norm: bool = False,
    ):    
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.time_dim = time_dim

        self.conv_1 = nn.Sequential(
            Norm(norm, input_dim),
            Activation(activation),
            nn.Conv2d(input_dim, self.output_dim, 3, padding=1)
        )
        self.conv_2 = nn.Sequential(
            Activation(activation),
            nn.Dropout(dropout),
            nn.Conv2d(self.output_dim, self.output_dim, 3, padding=1)
        )

        assert shortcut_kernel_size in [
            1, 3
        ], ('Only support `1` and `3` for `shortcut_kernel_size`, but '
            f'receive {shortcut_kernel_size}.')

        self.learnable_shortcut = self.output_dim != input_dim

        if self.learnable_shortcut:
            shortcut_padding = 1 if shortcut_kernel_size == 3 else 0
            self.shortcut = nn.Conv2d(
                input_dim,
                self.output_dim,
                shortcut_kernel_size,
                padding=shortcut_padding
            )

        if time_dim:
            self.norm_with_embedding = NormWithEmbedding(
                input_dim=self.output_dim,
                time_dim=time_dim,
                norm=norm,
                use_scale_shift=use_scale_shift_norm,
            )
        else:
            self.norm_with_embedding = lambda x, y: x

        self.init_weights()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.input_dim} -> {self.output_dim}: time_dim={self.time_dim})'
        
    def init_weights(self):
        nn.init.constant_(self.conv_2[-1].weight, 0)
        nn.init.constant_(self.conv_2[-1].bias, 0)

    def forward_shortcut(self, x):
        if self.learnable_shortcut:
            return self.shortcut(x)
        return x

    def forward(self, x: torch.Tensor, temb: Callable = None) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor ~ [batch_size, input_dim, h, w]): Input feature map tensor.
            temb (Callable): Dynamic function to obtain time embeddings.

        Returns:
            torch.Tensor: Output feature map tensor.
        """
        shortcut = self.forward_shortcut(x)
        x = self.conv_1(x)
        x = self.norm_with_embedding(x, temb)
        x = self.conv_2(x)
        return x + shortcut



