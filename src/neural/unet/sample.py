
from torch import nn
from torch.nn.functional import interpolate
import torch 


class Upsample(nn.Module):
    def __init__(
        self,
        input_dim: int,
        with_conv: bool = True,
        mode: str = 'nearest'
    ):
        """Upsampling operation used in the denoising network. Allows users to
        apply an additional convolution layer after the nearest interpolation operation.
        
        Args:
            input_dim (int): Number of channels of the input feature map to be downsampled.
            with_conv (bool, optional): Whether apply an additional convolution layer after
                upsampling.  Defaults to `True`.
        """
        super(Upsample, self).__init__()
        self.input_dim = input_dim 
        self.output_dim = input_dim 
        self.with_conv = with_conv
        self.mode = mode
        if with_conv:
            self.conv = nn.Conv2d(input_dim, input_dim, 3, 1, 1)
            
    def __repr__(self) -> str:
        return f'Upsample({self.input_dim}, with_conv={self.with_conv}, mode={self.mode})'

    def forward(self, x: torch.Tensor, shape: tuple[int, int] | None = None):
        """Forward function for upsampling operation.
        
        Args:
            x (torch.Tensor): Feature map to upsample.
            shape (tuple[int, int]): Output size.
            
        Returns:
            torch.Tensor: Feature map after upsampling.
        """
        x = interpolate(x, size=shape, mode=self.mode)
        if self.with_conv:
            x = self.conv(x)
        return x
    
    
class Downsample(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        with_conv: bool = True
    ):
        """Downsampling operation used in the denoising network. Support average
        pooling and convolution for downsample operation.

        Args:
            input_dim (int): Number of channels of the input feature map to be downsampled.
            with_conv (bool): Whether use convolution operation for 
                downsampling.  Defaults to `True`.
        """
        super(Downsample, self).__init__()
        self.input_dim = input_dim 
        self.output_dim = input_dim 
        self.with_conv = with_conv
        if with_conv:
            self.downsample = nn.Conv2d(input_dim, input_dim, 3, 2, 1)
        else:
            self.downsample = nn.AvgPool2d(stride=2)
            
    def __repr__(self) -> str:
        return f'Downsample({self.input_dim}, with_conv={self.with_conv})'

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        """Forward function for downsampling operation.
        
        Args:
            x (torch.Tensor): Feature map to downsample.

        Returns:
            torch.Tensor: Feature map after downsampling.
        """
        return self.downsample(x)
