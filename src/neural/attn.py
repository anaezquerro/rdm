from torch import nn 
from torch.nn.functional import softmax, dropout
from einops import rearrange
import torch 
from src.neural.norm import Norm 

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        num_heads: int = 8,
        dropout: float = 0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, 'Hidden dimension needs to be divisible by number of heads'
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.output_dim = output_dim or input_dim

        self.norm = Norm('layer', input_dim)
        self.Q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.K = nn.Linear(input_dim, hidden_dim, bias=False)
        self.V = nn.Linear(input_dim, hidden_dim, bias=False)
        self.scale = (hidden_dim//num_heads) ** -0.5
        self.out = nn.Linear(hidden_dim, self.output_dim) if hidden_dim != self.output_dim else nn.Identity()
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        if isinstance(self.out, nn.Linear):
            nn.init.xavier_uniform_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.input_dim}, {self.hidden_dim}, {self.output_dim})'

    def forward(self, x: torch.Tensor, *_, **__) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor ~ [batch_size, seq_len, input_dim]): Input source.

        Returns: 
            torch.Tensor ~ [batch_size, seq_len, output_dim].
        """
        x = self.norm(x)
        Q = rearrange(self.Q(x), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(self.K(x), 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(self.V(x), 'b n (h d) -> b h n d', h=self.num_heads)

        QK = torch.matmul(Q, K.transpose(-1,-2))
        scores = dropout(softmax(QK*self.scale, dim=-1), self.dropout, training=self.training)

        return self.out(rearrange(torch.matmul(scores, V), 'b h n d -> b n (h d)'))


class MultiHeadCrossAttention(nn.Module):
    """
    A multi-head cross-attention layer.
    """
    def __init__(
        self,
        input_dim1: int, 
        input_dim2: int,
        hidden_dim: int,
        output_dim: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, 'Hidden dimension needs to be divisible by number of heads'
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim1
        self.num_heads = num_heads
        self.dropout = dropout

        self.norm1 = Norm('layer', input_dim1)
        self.norm2 = Norm('layer', input_dim2)
        self.Q = nn.Linear(input_dim1, hidden_dim, bias=False)
        self.K = nn.Linear(input_dim2, hidden_dim, bias=False)
        self.V = nn.Linear(input_dim2, hidden_dim, bias=False)
        self.scale = (hidden_dim//num_heads)**-0.5
        self.out = nn.Linear(hidden_dim, self.output_dim) if hidden_dim != self.output_dim else nn.Identity()
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        if isinstance(self.out, nn.Linear):
            nn.init.xavier_uniform_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}([{self.input_dim1}, {self.input_dim2}] -> {self.hidden_dim} -> {self.output_dim})'

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x1 (torch.Tensor ~ [batch_size, seq_len1, input_dim1]): Query source.
            x2 (torch.Tensor ~ [batch_size, seq_len2, input_dim2]): Key/value source.

        Returns: 
            torch.Tensor ~ [batch_size, seq_len1, output_dim].
        """
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        Q = rearrange(self.Q(x1), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(self.K(x2), 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(self.V(x2), 'b n (h d) -> b h n d', h=self.num_heads)

        QK = torch.matmul(Q, K.transpose(-1,-2))
        scores = dropout(softmax(QK*self.scale, dim=-1), self.dropout, training=self.training)

        return self.out(rearrange(scores @ V, 'b h n d -> b n (h d)'))
    
        
