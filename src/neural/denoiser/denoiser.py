
from torch import nn 
from src.neural.embed import SinusoidalEmbedding
from src.util import Config 
import torch 
from torch.nn.functional import interpolate
from typing import Callable
from transformers import AutoModel
from torch import nn 
import torch 

from src.util import Config 
from src.neural.slot import SlotAutoEncoder, Abstractor
from src.neural.embed import PanelEmbedding

from src.env import RESULTS_FOLDER


class Denoiser(nn.Module):
    """General denoiser implementation (time and label conditioning)."""

    def build(self, time: Config):
        self.temb = SinusoidalEmbedding(**time)
        self.time_conf = time

    def cond(self, t: torch.Tensor) -> Callable:
        """Returns a function to compute the conditional signal.
        
        Args:
            t (torch.Tensor ~ [batch_size, 1, height, width]): Time signal.
            label (torch.Tensor ~ [batch_size, num_labels]): Class labels.

        Returns:
            Function that maps a size (h,w) -> [batch_size, height, wdith, cond_dim]
        """
        return lambda size: self.temb(interpolate(t, size=size, mode='nearest').squeeze(1)) 


class FeatureDenoiser(Denoiser):
    """Implementation of a denoiser conditioned with pretrained features."""

    def build(self, feature: Config):
        match feature.type:
            case 'pretrained':
                self.featurer = AutoModel.from_pretrained(feature.pretrained).requires_grad_(False)
            case 'slot':
                self.featurer = SlotAutoEncoder(**feature)
                self.featurer.load_state_dict(torch.load(f'{RESULTS_FOLDER}/{feature.path}', map_location='cuda')['model'])
                self.featurer = self.featurer.requires_grad_(False).eval()
            case _:
                raise ValueError(f'Unknown feature type: {feature.type}')

    def feat(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.featurer, SlotAutoEncoder):
            return self.featurer.slots(x)
        else:
            return self.featurer(x).last_hidden_state

class AbsDenoiser(Denoiser):
    """Relational Abstraction Denoiser, accepting slot configuration and building
    abstractors for slot-based CFG."""

    def build(self, slot: Config, rel: Config):
        self.slot_conf = slot 
        self.rel_conf = rel
        self.slot = SlotAutoEncoder(**slot)
        if slot.path:
            self.slot.load_state_dict(torch.load(f'{RESULTS_FOLDER}/{slot.path}', map_location='cuda')['model'])
        self.slot = self.slot.requires_grad_(False)
        self.slot.eval()
        self.abs = Abstractor(self.slot.hidden_dim, hidden_dim=rel.hidden_dim, num_layers=rel.num_layers)
        if len(rel.inputs) > 1:
            self.panel = PanelEmbedding(len(rel.inputs), self.slot.hidden_dim)
        else:
            self.panel = nn.Identity()
        if isinstance(rel.pdrop.training, list):
            self.default_feat = nn.Parameter(torch.randn( slot.num_slots, self.slot.encoder.hidden_dim))
            self.default_pos = nn.Parameter(torch.randn( slot.num_slots, self.slot.encoder.hidden_dim))
            rel.pdrop.training = torch.tensor(rel.pdrop.training)
            rel.pdrop.inference = torch.tensor(rel.pdrop.inference)
        else:
            self.default = nn.Parameter(torch.randn(slot.num_slots, rel.hidden_dim))

    def drop(self, x: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
        """Drop embeddings to simulate CFG.
        
        Args:
            rel (torch.Tensor ~ [batch_size, num_inputs, num_slots, dim]): Embeddings.
            default (torch.Tensor ~ [num_slots, dim]): Default embeddings.

        Returns:
            torch.Tensor ~ [batch_size, num_inputs, num_slots, dim]: Dropped embeddings.
        """
        b = x.shape[0]
        if self.training:
            ndrop = b*self.rel_conf.pdrop.training
        else:
            ndrop = b*self.rel_conf.pdrop.inference
        indices = torch.randperm(b)
        if isinstance(ndrop, torch.Tensor):
            ndrop = ndrop.int()
            mask = torch.ones(b, len(self.rel_conf.inputs), dtype=bool, device=x.device)
            for i in range(len(self.rel_conf.inputs)):
                mask[indices[:ndrop[i]], i] = False
        else:
            ndrop = int(ndrop)
            mask = torch.ones(b, dtype=bool, device=x.device)
            mask[indices[:ndrop]] = False 
        xdrop = torch.where(mask.view(b, -1, 1, 1), x, default)
        return xdrop

    def rel_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relational embeddings.

        Args:
            x (torch.Tensor ~ [batch_size, input_dim, height, width]): Inputs to the slot autoencoder.
            pdrop (float): Probability of removing specific input channels.
            mask (torch.Tensor ~ [batch_size, num_inputs]): Channels to maintain (True) or drop (False).

        Returns:
            torch.Tensor ~ [batch_size, num_inputs*num_slots, cond_dim]: Relational embeddings.
            torch.Tensor ~ [batch_size, input_dim, height, width]: Slot AE reconstruction.
        """
        b = x.shape[0]
        _, feat, pos = zip(*map(self.slot.get, x.split(self.slot.num_channels, dim=1)))
        pos = self.panel(torch.stack(pos, dim=1))
        feat = self.panel(torch.stack(feat, dim=1))
        if isinstance(self.rel_conf.pdrop.training, torch.Tensor):
            feat = self.drop(feat, self.default_feat)
            pos = self.drop(pos, self.default_pos)
        rel = self.abs(feat.flatten(1,2), pos.flatten(1,2))
        if not isinstance(self.rel_conf.pdrop.training, torch.Tensor):
            rel = self.drop(rel.reshape(b, len(self.rel_conf.inputs), self.slot.num_slots, -1), self.default).flatten(1,2)
        return rel