from torch import nn 
from torch.nn.functional import interpolate
import torch

class Interpolate(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        size: tuple[int, int] | int,
        output_dim: int = None,
        mode: str = 'bilinear',
        use_conv: bool = True
    ):
        super(Interpolate, self).__init__()
        self.input_dim = input_dim 
        self.size = size if isinstance(size, tuple) else (size, size)
        self.output_dim = output_dim or input_dim 
        self.mode = mode
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(input_dim, self.output_dim, 3, 1, 1)
        else:
            self.conv = nn.Identity()
        self.reset_parameters()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size}, dim={self.output_dim}, use_conv={self.use_conv})'

    def reset_parameters(self):
        if self.use_conv:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(interpolate(x.contiguous(), size=self.size, mode=self.mode, align_corners=False))
    