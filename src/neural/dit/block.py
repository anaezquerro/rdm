
from torch import nn 
from timm.models.vision_transformer import Attention, Mlp
import torch 

from src.neural.interp import Interpolate

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(
        self,
        hidden_size: tuple[int, int, int], 
        output_size: tuple[int, int, int],
        num_layers: int,
        kernel_size: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_size[0]
        self.hidden_size = hidden_size
        self.norm_final = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim, bias=True)
        )
        layers = []
        for _ in range(num_layers):
            layers += [
                nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size),
                nn.GroupNorm(32, self.hidden_dim),
                nn.ReLU()
            ]
        layers.append(Interpolate(self.hidden_dim, output_size[1:], output_dim=output_size[0]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.layers(x.permute(0,2,1).reshape(-1, *self.hidden_size))
        return x
