import torch 
from torch import nn

from .denoiser import Denoiser, FeatureDenoiser, AbsDenoiser
from src.util import Config 

from src.neural.unet import UNet

class DenoiserUNet(UNet, Denoiser):
    
    def __init__(self, time: Config, unet: Config):
        super().__init__(**unet, time_dim=time.output_dim)
        Denoiser.build(self, time)
        
    def forward(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor, 
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        pos: torch.Tensor = None,
        feat: torch.Tensor = None,
        **_
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            xt (torch.Tensor ~ [batch_size, input_dim, height, width]): Input noise.
            t (torch.Tensor ~ [batch_size, 1, height, width]): Timesteps.
            feat (torch.Tensor ~ [batch_size, num_feats, cond_dim]): Hidden features.
            
        Returns: 
            torch.Tensor ~ [batch_size, output_dim, height, width]: Prediction.
        """
        temb = self.cond(t)
        h, hs = self.encoder(xt, temb=temb, key=key, value=value, pos=pos, feat=feat)
        h = self.bottleneck(h, temb=temb, key=key, value=value, pos=pos, feat=feat)
        out = self.decoder(h, hs, temb=temb, key=key, value=value, pos=pos, feat=feat)
        return out 
    
    
class AbsUNet(DenoiserUNet, AbsDenoiser):
    def __init__(self, slot: Config, rel: Config, unet: Config, **kwargs):
        unet.attention.feat_dim = rel.hidden_dim 
        unet.attention.pos_dim = slot.encoder.hidden_dims[-1]
        super().__init__(unet=unet, **kwargs)
        AbsDenoiser.build(self, slot, rel) 
        if unet.shared_attn:
            self.K = nn.Linear(rel.hidden_dim, rel.hidden_dim, bias=False)
            self.V = nn.Linear(rel.hidden_dim, rel.hidden_dim, bias=False)
        else:
            self.K, self.V = nn.Identity(), nn.Identity()
        
    def forward(self, slot: torch.Tensor, **_) -> torch.Tensor:
        """
        Forward pass. Adds relational embeddings as features for the standard U-Net forward pass.
        
        Args:
            slot (torch.Tensor ~ [batch_size, input_dim, height, width]): Input to the slot autoencoder.
            mask (torch.Tensor ~ [batch_size, num_inputs]): Mask to remove input signals.

        Returns: 
            torch.Tensor ~ [batch_size, output_dim, height, width]: Prediction.
            torch.Tensor ~ [batch_size, input_dim, height, width]: Slot autoencoder reconstruction.
        """
        rel = self.rel_embed(slot)
        key, value = self.K(rel), self.V(rel)
        pos = self.slot.positions
        return super(AbsUNet, self).forward(**_, feat=rel, key=key, value=value, pos=pos)
        

class FeatureUNet(DenoiserUNet, FeatureDenoiser):
    def __init__(self, unet: Config, feature: Config, time: Config):
        unet.attention.feat_dim = feature.dim
        super().__init__(unet=unet, time=time)
        FeatureDenoiser.build(self, feature)
        
    def forward(self, feat: torch.Tensor, **_) -> torch.Tensor:
        feat = self.feat(feat)
        return super(FeatureUNet, self).forward(**_, feat=feat)

        