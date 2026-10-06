from torch import nn 
from math import prod
import torch

from .attn import SlotAttention
from src.util import Config
from src.neural.embed import SoftGridEmbedding
from .encoder import SlotEncoder
from .decoder import SlotDecoder, ScalableDecoder

class SlotAutoEncoder(nn.Module):
    def __init__(
        self, 
        img_size: tuple[int, int, int],
        num_slots: int, 
        iters: int, 
        hidden_size: tuple[int, int, int],
        encoder: Config,
        decoder: Config,
        **_
    ):
        """Adaptation of the Slot AutoEncoder ([Locatello et al., 2020](https://arxiv.org/abs/2006.15055))

        Args:
            img_size (tuple[int, int, int]): Image dimensions, [C, H, W].
            num_slots (int): Number of slots.
            iters (int): Number of Slot Attention iterations.
            hidden_size (tuple[int, int, int]): Hidden resolution for reconstruction, [D, H', W'].
        """
        super().__init__()
        self.img_size = img_size
        self.num_channels, *self.resolution = img_size 
        self.num_slots = num_slots
        self.iters = iters
        self.hidden_dim = hidden_size[0]
        self.hidden_size = hidden_size
        
        # create pretrained encoder
        self.encoder = SlotEncoder(img_size, **encoder)
        self.num_feats = prod(self.encoder.output_size[1:])
            
        self.pemb = SoftGridEmbedding(self.encoder.hidden_dim)

        # slot-attention layer
        self.attn = SlotAttention(num_slots, self.encoder.hidden_dim, self.hidden_dim, iters, eps=1e-8)

        # create the decoder
        if decoder.scalable:
            self.decoder = ScalableDecoder(img_size, hidden_size, **decoder, add_mask=False)
        else:
            self.decoder = SlotDecoder(img_size, hidden_size, **decoder, add_mask=True)


    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            img (torch.Tensor ~ [batch_size, input_dim, height, width]): Input images.

        Returns:
            torch.Tensor ~ [batch_size, input_dim, height, width]: Reconstructed images.
        """
        # feat ~ [batch_size, num_feats, hidden_dim]
        feat = self.encoder(img)
        pos = self.positions(self.encoder.output_size[1:], img.device).flatten(-2,-1).movedim(0,-1).unsqueeze(0)
        
        # slots, feat, pos ~ [batch_size, num_slots, hidden_dim]
        slots, *_ = self.attn(feat + pos, feat, pos)
        recons, *_ = self.decoder(slots) 
        return recons

    def slots(self, img: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(img)
        pos = self.positions(self.encoder.output_size[1:], img.device).flatten(-2,-1).movedim(0,-1).unsqueeze(0)
        slots, *_ = self.attn(feat + pos, feat, pos)
        return slots 

    def positions(self, resolution: tuple[int, int], device: torch.device):
        """Downstream positional embeddings.
        
        Args:
            resolution (tuple[int, int]): Target resolution.
            device (torch.device): CUDA device.

        Returns:
            torch.Tensor ~ [dim, *resolution]
        """
        if self.hard_pos:
            return self.pemb.reshape(*self.encoder.output_size[1:], -1).movedim(-1, 0)
        else:
            return self.pemb(resolution, device)
    
    def disentangle(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the slot disentanglement.

        Args:
            imgs (torch.Tensor ~ [batch_size, input_dim, height, width]): Input images.

        Returns:
            torch.Tensor ~ [batch_size, input_dim, height, width]: Reconstructed images.
            torch.Tensor ~ [batch_size, num_slots, input_dim, height, width]: Reconstruction per slot.
            torch.Tensor ~ [batch_size, num_slots, 1, height, width]: Attention scores per slot.
        """
        # feat ~ [batch_size, num_feats, hidden_dim]
        feat = self.encoder(img)
        pos = self.positions(self.encoder.output_size[1:], img.device).flatten(-2,-1).movedim(0,-1).unsqueeze(0)
        
        # slots, feat, pos ~ [batch_size, num_slots, hidden_dim]
        slots, *_ = self.attn(feat + pos, feat, pos)
        return self.decoder(slots) 

    def get(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gets slots, features and position embeddings.

        Args:
            imgs (torch.Tensor): Input map feature.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Slot, features and position embeddings.
        """
        feat = self.encoder(img)
        pos = self.positions(self.encoder.output_size[1:], img.device).flatten(-2,-1).movedim(0,-1).unsqueeze(0)
        slots, feat, pos = self.attn(feat+pos, feat, pos)
        return slots.detach(), feat.detach(), pos.detach()
    