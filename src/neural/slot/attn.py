from torch import nn
import torch
from torch.nn.functional import softmax, relu

class SlotAttention(nn.Module):
    def __init__(
        self,
        num_slots: int, 
        input_dim: int,
        hidden_dim: int, 
        iters: int = 3, 
        eps = 1e-8,
        **_
    ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = hidden_dim ** -0.5
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim

        self.slots_mu = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, hidden_dim)))

        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(input_dim, hidden_dim)
        self.to_v = nn.Linear(input_dim, hidden_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.norm_input  = nn.LayerNorm(input_dim)
        self.norm_slots  = nn.LayerNorm(hidden_dim)
        self.norm_pre_ff = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, feat: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Apply slot-attention.

        Args:
            x (torch.Tensor ~ [batch_size, num_feats, input_dim]): Image embeddings.
            feat (torch.Tensor ~ [batch_size, num_feats, input_dim]): Feature embeddings.
            pos (torch.Tensor ~ [batch_size, num_feats, input_dim]): Positional embeddings.
        """
        b = x.shape[0]
        d = self.hidden_dim
		
        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_sigma.expand(b, self.num_slots, -1)
        slots = torch.normal(mu, sigma)

        x = self.norm_input(x) 
        pos = pos.expand(b,pos.shape[1],pos.shape[2])
        k, v = self.to_k(x), self.to_v(x)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = softmax(dots.to(torch.float32), dim=1).to(dots.dtype) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(relu(self.fc1(self.norm_pre_ff(slots))))

        return slots, torch.einsum('bjd,bij->bid', feat, attn), torch.einsum('bjd,bij->bid', pos, attn)
