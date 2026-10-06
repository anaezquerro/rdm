from torch import nn 
import torch 

from src.neural.act import Activation

class FFN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        act: str = 'leakyrelu',
        dropout: float = 0.
    ): 
        super().__init__()
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.act = act
        self.dropout = dropout

        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            Activation(act),
            nn.Dropout(dropout)
        )
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.proj[0].weight)
        nn.init.constant_(self.proj[0].bias, 0.0)

    def forward(self, x: torch.Tensor):
        return self.proj(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.input_dim}, {self.output_dim}' + f', {self.act}'*(self.act != 'identity') + f', dropout={self.dropout})'
