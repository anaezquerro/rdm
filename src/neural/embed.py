from torch import nn
from math import log
import torch 

from src.neural.act import Activation

class Embedding(nn.Module): 
    def __init__(
        self,
        embed_dim: int = 512, 
        output_dim: int = 512, 
        act: str = 'silu',
        **_
    ):
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim 
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            Activation(act), 
            nn.Linear(output_dim, output_dim)
        )
    
    def embed(self, x: torch.Tensor) -> torch.Tensor: 
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(x))


class SinusoidalEmbedding(Embedding): 
    
    def __init__(
        self,
        scale: float | int = 1000,
        shift: float | int = -1,
        max_period: int = 10000,
        **_
    ):
        super(SinusoidalEmbedding, self).__init__(**_)
        self.scale = scale 
        self.shift = shift 
        self.neg_log_max_period = -log(max_period)
        
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(
            self.neg_log_max_period *
            torch.arange(start=0, end=half, dtype=torch.float32, device=x.device) /
            half
        ).to(x.dtype)
        args = torch.clamp_min(self.scale * x.unsqueeze(-1) + self.shift, 0) * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embed_dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding


class SoftGridEmbedding(nn.Module):
    def __init__(self, dim: int):
        """Implementation of the grid embeddings.
        
        Args:
            dim (int): Embedding dimension.
        """
        super().__init__()
        self.dim = dim 
        self.embed = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, resolution: tuple[int, int], device: torch.device) -> torch.Tensor:
        """Forward pass.
        
        Args:
            device (torch.device): Current CUDA device.

        Returns:
            torch.Tensor ~ [dim, *resolution]: Soft grid embeddings.
        """
        rows = torch.linspace(-1, 1, resolution[0], device=device)
        cols = torch.linspace(-1, 1, resolution[1], device=device)
        grid = self.embed(torch.stack(torch.meshgrid(rows, cols, indexing='ij'), dim=-1).flatten(0,1)).reshape(*resolution, -1)
        return grid.permute(2, 0, 1)



class SoftPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, num_freqs: int = 10):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = torch.exp(torch.linspace(0, log(1000.0), num_freqs))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(num_freqs * 2, dim)

    def forward(self, x: torch.Tensor):
        """Forward pass.
        
        Args:
            x (torch.Tensor ~ [*shape, seq_len, dim]): Input features.

        Returns:
            torch.Tensor ~ [*shape, seq_len, dim]:y
        """
        pos = torch.linspace(-1, 1, x.shape[-2], device=x.device).view(-1, 1)
        angles = pos * self.freqs.view(1, -1)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        emb = self.proj(emb).unsqueeze(0)
        return x + emb


class PanelEmbedding(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        hidden_dim: int 
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.emb = nn.Parameter(torch.randn(num_inputs, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor ~ [batch_size, num_panels, seq_len, hidden_dim]): Input to contextualize.
            
        Returns: 
            torch.Tensor ~ [batch_size, num_panels, seq_len, hidden_dim].
        """
        return x + self.emb.view(1, self.num_inputs, 1, -1)