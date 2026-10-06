
from torch import nn 
from einops import rearrange
from torch.nn.functional import softmax, dropout
import torch 

from src.neural.norm import Norm
from src.neural.attn import MultiHeadSelfAttention
from src.neural.ffn import FFN 

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class RelationalAttention(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64,
        output_dim: int | None = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.scale = hidden_dim ** -0.5

        self.norm = nn.LayerNorm(input_dim)
        torch.manual_seed(3+layer_idx)
        self.Q = nn.Linear(input_dim, hidden_dim*num_heads, bias=False)
        torch.manual_seed(3+layer_idx)
        self.K = nn.Linear(input_dim, hidden_dim*num_heads, bias=False)
        self.V = nn.Linear(input_dim, hidden_dim*num_heads, bias=False)
        self.out = FFN(hidden_dim*num_heads, self.output_dim, act='linear', dropout=dropout)
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.input_dim}, {self.hidden_dim}, {self.output_dim})'

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        x = self.norm(x)
        s = self.norm(s)
        
        Q = rearrange(self.Q(x), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(self.K(x), 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(self.V(s), 'b n (h d) -> b h n d', h=self.num_heads)

        QK = torch.matmul(Q, K.transpose(-1, -2))
        scores = dropout(softmax(QK*self.scale, dim=-1), self.dropout, training=self.training)
        return self.out(rearrange(torch.matmul(scores, V), 'b h n d -> b n (h d)'))

class Abstractor(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        num_layers: int, 
        hidden_dim: int = None, 
        num_heads: int = 8, 
        dropout: float = 0,
        output_dim: int | None = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim or input_dim
        self.output_dim = output_dim or self.hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.feat = nn.Linear(input_dim, self.hidden_dim) if input_dim != self.hidden_dim else nn.Identity()
        self.pos = nn.Linear(input_dim, self.hidden_dim) if input_dim != self.hidden_dim else nn.Identity()
        for l_idx in range(num_layers):
            block = nn.ModuleList([
                RelationalAttention(self.hidden_dim, self.hidden_dim, num_heads=num_heads, dropout=dropout, layer_idx=l_idx),
                MultiHeadSelfAttention(self.hidden_dim, self.hidden_dim, num_heads=num_heads, dropout=dropout),
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                )
            ])
            self.layers.append(block)
        self.out = nn.Linear(self.hidden_dim, self.output_dim) if self.hidden_dim != self.output_dim else nn.Identity()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.input_dim}, {self.hidden_dim}, num_layers={self.num_layers})'

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.feat(x)
        s = self.pos(s)
        for rel_attn, self_attn, ff in self.layers:
            s = rel_attn(x,s) + s
            s = ff(s) + s
            s = self_attn(s)+ s
            s = ff(s) + s
        return self.out(s)
    
    
    