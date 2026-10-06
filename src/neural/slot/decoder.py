
from torch import nn 
import torch
from torch.nn.functional import softmax, interpolate
from src.neural.interp import Interpolate
from src.neural.conv import ConvBlock
from src.neural.embed import SoftGridEmbedding


class SlotDecoder(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int, int],
        hidden_size: tuple[int, int, int],
        hidden_dims: list[int], 
        num_convs: int,
        add_mask: bool,
        **_
    ):
        """Implementation of the Slot Decoder.
        
        Args:
            img_size (tuple[int, int, int]): Original image size, [C, H, W].
            hidden_size (tuple[int, int, int]): Resolution from which to start the reconstruction.
            hidden_dims (list[int]): Hidden dimensions for the convolutional blocks.
            num_convs (int): Number of convolutions per block.
            add_mask (bool): Wether to add an extra channel for slot scores.
        """
        super(SlotDecoder, self).__init__()
        self.img_size = img_size
        self.output_dim, *self.resolution = img_size 
        self.hidden_dim = hidden_size[0]
        self.hidden_size = hidden_size
        self.hidden_dims = hidden_dims.copy()
            
        layers = []
        target_size = max(self.resolution)
        dim = self.hidden_dim
        size = max(hidden_size[1:])
        while size < target_size:
            next_size = min(target_size, size*2) if len(hidden_dims) > 1 else target_size
            layers += [
                ConvBlock(
                    input_dim=dim, 
                    output_dim=hidden_dims[0],
                    kernel_size=5,
                    num_convs=num_convs,
                    norm=None,
                    padding=2
                ),
                Interpolate(hidden_dims[0], size=next_size, use_conv=True),
                ]
            size = next_size
            dim = hidden_dims.pop(0)
        layers.append(nn.Conv2d(dim, self.output_dim+add_mask, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)
        self.pemb = SoftGridEmbedding(hidden_size[0])


    def forward(self, slot: torch.Tensor)  -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            slot (torch.Tensor ~ [batch_size, num_slots, hidden_dim]): Slot embeddings.

        Returns:
            torch.Tensor ~ [batch_size, output_dim, height, width]: Reconstructed image.
            torch.Tensor ~ [batch_size, num_slots, output_dim, height, width]: Reconstructed image per slot.
            torch.Tensor ~ [batch_size, num_slots, 1, height, width]: Attention mask.
        """
        b, k = slot.shape[:2]
        # broadcast slots into the hidden size
        pemb = self.pemb(resolution=self.hidden_size[1:], device=slot.device).unsqueeze(0)
        x = slot.view(-1, slot.shape[-1], 1, 1).expand(-1, -1, *self.hidden_size[1:]) + pemb
        recons, mask = self.layers(x).split([self.output_dim, 1], dim=1)
        mask = mask.view(b, k, 1, *self.resolution)
        recons = recons.view(b, k, *self.img_size)
        attn = softmax(mask.to(torch.float32), dim=1).to(slot.dtype)
        return (recons * attn).sum(dim=1), recons, attn 
    

class ScalableDecoder(SlotDecoder):
    """Implementation of the scalable decoder."""
        
    def forward(self, slot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            slot (torch.Tensor ~ [batch_size, num_slots, hidden_dim]): Slot embeddings.

        Returns:
            torch.Tenosr ~ [batch_size, output_dim, height, width]: Reconstructed image.
            torch.Tensor ~ [batch_size, num_slots, output_dim, height, width]: Reconstructed image per slot (only for inference).
            torch.Tensor ~ [batch_size, num_slots, 1, height, width]: Attention mask.
        """
        b, k = slot.shape[:2]

        # compute the attention for competition
        # [batch_size, num_slots, num_feats]
        attn_weights = torch.einsum('bsd,df -> bsf', slot, self.pemb(self.hidden_size[1:], slot.device).flatten(-2,-1))
        attn_weights = softmax(attn_weights.to(torch.float32), dim=1).to(slot.dtype)
        
        recons = self.reconstruct(attn_weights, slot)
        if not self.training:
            slot_recons = []
            for slot in slot.split(1, dim=1):
                slot_recons.append(
                    self.reconstruct(
                        attn_weights=torch.ones(b, 1, attn_weights.shape[-1], device=slot.device),
                        slots=slot
                    )
                )
            slot_recons = torch.stack(slot_recons, dim=1)
            attn_weights = attn_weights.view(b, -1, *self.hidden_size[1:])
            attn_weights = interpolate(attn_weights, size=self.resolution, mode='nearest')
        else:
            slot_recons = None 
        return recons, slot_recons, attn_weights

    def reconstruct(self, attn_weights: torch.Tensor, slots: torch.Tensor):
        """Reconstructs the image based on the scores and slots.
        
        Args:
            attn_weights (torch.Tensor ~ [batch_size, num_slots, num_feats]): Attention weights to the features.
            slots (torch.Tensor ~ [batch_size, num_slots, dim]): Slot embeddings.
            
        Returns:
            torch.Tensor ~ [batch_size, output_dim, height, width]: Reconstructed image.
        """
        context = torch.matmul(attn_weights.transpose(1,2), slots)
        context = context.permute(0,2,1).view(attn_weights.shape[0], *self.hidden_size)
        recons = self.layers(context)
        return recons
    
    