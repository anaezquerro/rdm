
from torch import nn 
import torch 

from .act import Activation
from .norm import Norm


class ConvLayer(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        output_dim: int, 
        kernel_size: int,
        act: str = 'relu',
        norm: str = 'group',
        **_
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size 
        self.act = act
        self.norm = norm
        
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, bias=norm is None, **_)
        self.norm = Norm(norm, output_dim)
        self.act = Activation(act)
        self.reset_parameters()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.input_dim}, {self.output_dim}, {self.kernel_size}, stride={self.conv.stride}, padding={self.conv.padding}, norm={self.norm}, act={self.act})'

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))

class ConvBlock(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        num_convs: int = 1,
        **params
    ):
        layers = [ConvLayer(input_dim, output_dim, kernel_size, **params)]
        for _ in range(num_convs-1):
            layers.append(ConvLayer(output_dim, output_dim, kernel_size, **params))
        super().__init__(*layers)

class AddCoords(nn.Module):
    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        row = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        col = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, row, col], dim=1)

class CoordConv(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__()
        self.addcoords = AddCoords()
        self.conv = nn.Conv2d(in_channels + 2, *args, **kwargs)

    def forward(self, x):
        return self.conv(self.addcoords(x))