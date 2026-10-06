import torch
from torch import nn
from einops import rearrange
from torch.nn.functional import softmax, interpolate

from src.neural.norm import Norm
from typing import Callable


class MultiHeadSpatialAttention(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 1,  # will be ignored if num_head_channels is set
        num_head_channels: int | None = None,
        norm: str = 'group',
        **_
    ):
        """An attention block allows spatial position to attend to each other.

        Originally ported from here, but adapted to the N-d case.
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.  # noqa

        Args:
            input_dim (int): Channels of the input feature map.
            num_heads (int): Number of heads in the attention.
            norm (dict, optional): Config for normalization layer. Defaults to GroupNorm with 32 groups.
        """
        super(MultiHeadSpatialAttention, self).__init__()
        if num_head_channels is None:
            self.num_heads = num_heads
        else:
            assert input_dim % num_head_channels == 0
            self.num_heads = input_dim // num_head_channels
        self.norm = Norm(norm, input_dim)
        self.qkv = nn.Conv1d(input_dim, input_dim * 3, 1)
        self.proj = nn.Conv1d(input_dim, input_dim, 1)
        self.init_weights()
        self.input_dim, self.output_dim = input_dim, input_dim

    def init_weights(self):
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.input_dim}, {self.num_heads})'


    @staticmethod
    def QKVAttention(qkv):
        channel = qkv.shape[1] // 3
        q, k, v = torch.chunk(qkv, 3, dim=1)
        scale = 1 / channel ** 0.25
        weight = torch.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        weight = torch.einsum('bts,bcs->bct', weight, v)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for multi head attention.
        
        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Feature map after attention.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.QKVAttention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj(h)
        return (h + x).reshape(b, c, *spatial)



class MultiHeadSharedCrossAttention(nn.Module):

    def __init__(
        self,
        input_dim: int,
        feat_dim: int,
        pos_dim: int,
        num_heads: int = 1,
        num_head_channels: int | None = None,
        norm: str = 'group'
    ):
        """Implementation of the multi-head cross attention module
        to mix hidden maps with hidden features.

        Args:
            input_dim1 (int): Dimension of the hidden feature map.
            input_dim2 (int): Dimension of the hidden features.
        """
        super(MultiHeadSharedCrossAttention, self).__init__()
        self.input_dim = input_dim 
        self.feat_dim = feat_dim 
        
        if num_head_channels is None:
            assert feat_dim % num_heads == 0
            self.num_heads = num_heads
        else:
            assert feat_dim % num_head_channels == 0
            self.num_heads = feat_dim // num_head_channels
        self.norm = Norm(norm, input_dim)
        self.pos = nn.Linear(pos_dim, input_dim)
        self.Q = nn.Linear(input_dim, feat_dim, bias=False)
        self.out = nn.Linear(feat_dim, input_dim)

    def init_weights(self):
        nn.init.constant_(self.pos.weight, 0)
        nn.init.constant_(self.pos.bias, 0)
        nn.init.constant_(self.out.weight, 0)
        nn.init.constant_(self.out.bias, 0)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.input_dim}, {self.feat_dim}, {self.num_heads})'

    def forward(self, x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, pos: Callable, **_) -> torch.Tensor:
        """Forward function for multi head cross attention.
        
        Args:
            x (torch.Tensor ~ [batch_size, input_dim, height, width]): Input feature map.
            feat (torch.Tensor ~ [batch_size, num_feats, feat_dim]): Input features.
            pos (torch.Tensor ~ [*grid, dim]): Callable por positional embeddings.

        Returns:
            torch.Tensor: Feature map after attention.
        """
        # normalize and add positional features
        pemb = self.pos(pos(x.shape[-2:], x.device).movedim(0,-1)).movedim(-1,0).unsqueeze(0)
        xpos = interpolate(pemb, size=x.shape[-2:], mode='nearest') + self.norm(x)
        xpos = xpos.flatten(-2,-1).movedim(1,-1) # [batch_size, height * width, dim]

        # attention scores
        Q = rearrange(self.Q(xpos), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(key, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(value, 'b n (h d) -> b h n d', h=self.num_heads)
        
        QK = torch.matmul(Q, K.transpose(-1,-2))
        scale = Q.shape[-1]**-0.5
        scores = softmax(QK*scale, dim=-1)
        
        # h ~ [batch_size, input_dim, height*width]
        h = self.out(rearrange(scores @ V, 'b h n d -> b (h d) n').permute(0,2,1))
        return h.reshape(x.shape) + x





class MultiHeadSpatialCrossAttention(nn.Module):

    def __init__(
        self,
        input_dim: int,
        feat_dim: int,
        num_heads: int = 1,
        num_head_channels: int | None = None,
        norm: str = 'group',
        **_
    ):
        """Implementation of the multi-head cross attention module
        to mix hidden maps with hidden features.

        Args:
            input_dim1 (int): Dimension of the hidden feature map.
            input_dim2 (int): Dimension of the hidden features.
        """
        super(MultiHeadSpatialCrossAttention, self).__init__()
        self.input_dim = input_dim 
        self.feat_dim = feat_dim 
        self.output_dim = input_dim
        
        if num_head_channels is None:
            assert input_dim % num_heads == 0
            self.num_heads = num_heads
        else:
            assert input_dim % num_head_channels == 0
            self.num_heads = input_dim // num_head_channels
        self.norm = Norm(norm, input_dim)
        self.Q = nn.Linear(input_dim, input_dim, bias=False)
        self.K = nn.Linear(feat_dim, input_dim, bias=False)
        self.V = nn.Linear(feat_dim, input_dim, bias=False)
        self.proj = nn.Linear(input_dim, input_dim)

    def init_weights(self):
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.input_dim}, {self.feat_dim}, {self.num_heads})'

    def forward(self, x: torch.Tensor, feat: torch.Tensor, **_) -> torch.Tensor:
        """Forward function for multi head cross attention.
        
        Args:
            x (torch.Tensor ~ [batch_size, input_dim, height, width]): Input feature map.
            feat (torch.Tensor ~ [batch_size, num_feats, feat_dim]): Input features.

        Returns:
            torch.Tensor: Feature map after attention.
        """
        # reshape hidden map 
        b, c, *_ = x.shape
        xnorm = self.norm(x.reshape(b, c, -1)).permute(0,2,1) # [batch_size, height*width, input_dim]
        
        Q = rearrange(self.Q(xnorm), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(self.K(feat), 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(self.V(feat), 'b n (h d) -> b h n d', h=self.num_heads)
        
        QK = torch.matmul(Q, K.transpose(-1,-2))
        scale = self.input_dim**-0.5
        scores = softmax(QK*scale, dim=-1)
        
        # h ~ [batch_size, input_dim, height*width]
        h = self.proj(rearrange(scores @ V, 'b h n d -> b (h d) n').permute(0,2,1))
        return h.reshape(x.shape) + x