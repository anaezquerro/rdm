from torch import nn 
import torch 

from src.neural.conv import ConvBlock
from src.neural.interp import Interpolate

class SlotEncoder(nn.Module):
    def __init__(
        self, 
        img_size: tuple[int, int, int], 
        hidden_dims: list[int], 
        num_convs: int,
        norm: str = 'group',
        **_
    ):
        """Implementation of a convolutional encoder.

        Args:
            img_size (tuple[int, int, int]): Original image dimensions, [C, H, W].
            hidden_dims (list[int]): List of hidden dimensions (number of channels).
            num_convs (int): Number of convolutions per layer.
        """
        super().__init__()
        self.img_size = img_size
        self.hidden_dim = hidden_dims[-1]
        self.hidden_dims = hidden_dims

        layers = [nn.Conv2d(img_size[0], hidden_dims[0], kernel_size=3, padding=1)] 
        size = min(img_size[1:])
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            layers += [
                ConvBlock(in_dim, out_dim, kernel_size=5, padding=2, num_convs=num_convs, norm=norm),
                Interpolate(out_dim, size=size//2, use_conv=False)
            ]
            size //= 2 
            in_dim = out_dim 
        self.output_size = (self.hidden_dim, size, size)
        self.layers = nn.Sequential(*layers)
    
    @property
    def num_feats(self) -> int:
        return self.output_size[1]*self.output_size[2]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x) 
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        return x

