from __future__ import annotations 
from typing import Callable
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt 
import torch

from src.neural import SlotAutoEncoder, UNet
from src.util import postprocess
from .trainer import Trainer
from .opt import Optimizer


class AutoEncoderTrainer(Trainer):
    NAME = 'ae'
            
    def train_step(self, input: torch.Tensor, target: torch.Tensor, **__) -> torch.Tensor:
        return dict(loss=mse_loss(self.model(input), target))
    
    @torch.no_grad()
    def pred_step(self, input: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        return dict(pred=self.model(input))
    
    def save_pred(
        self, 
        step: int,
        path: str,
        pred: torch.Tensor, 
        target: torch.Tensor,
        **_
    ):
        for i, (pred, target) in enumerate(zip(pred.unbind(0), target.unbind(0))):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(postprocess(pred))
            ax[0].title.set_text('Predicted')
            ax[1].imshow(postprocess(target))
            ax[1].title.set_text('Real')
            for a in ax.flatten():
                a.axis('off')
            fig.savefig(f'{path}/{step}-{i}.jpg', bbox_inches='tight')
            plt.close()

    @classmethod
    def from_args(cls, args, data: Callable) -> AutoEncoderTrainer:
        model = UNet(
            input_dim=sum(data.NUM_CHANNELS[name] for name in args.conf.data.transform.input),
            output_dim=sum(data.NUM_CHANNELS[name] for name in args.conf.data.transform.target),
            resolution=args.conf.data.resolution,
            **args.conf.unet
        )
        return cls(
            model=model,
            opt=Optimizer.from_conf(model, **args.conf.optimizer),            
            device=args.device,
            dtype=args.dtype
        )
    
    
    
class SlotAutoEncoderTrainer(AutoEncoderTrainer):
    NAME = 'sae'
            
    @torch.no_grad()
    def pred_step(self, input: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        pred, recon, attn = self.model.disentangle(input)
        return dict(pred=pred, recon=recon, attn=attn)
    
    def save_pred(
        self, 
        step: int,
        path: str,
        pred: torch.Tensor, 
        recon: torch.Tensor, 
        attn: torch.Tensor,
        target: torch.Tensor,
        **_
    ):
        num_slots = recon.shape[1]
        for i, (pred, target) in enumerate(zip(pred.unbind(0), target.unbind(0))):
            fig, ax = plt.subplots(1, num_slots+2, figsize=((num_slots+2)*2, 2))
            ax[0].imshow(postprocess(target))
            # ax[0].title.set_text(f'Real')
            ax[1].imshow(postprocess(pred))
            # ax[1].title.set_text(f'Reconstructed')
            for k in range(num_slots):
                # ax[k+2].imshow(postprocess(attn[i,k], mean=0, std=1))
                ax[k+2].imshow(postprocess(recon[i,k] - 1*attn[i,k]))
                # ax[k+2].title.set_text(f'Slot {k+1}')
            for a in ax.flatten():
                a.axis('off')
            fig.subplots_adjust(
                wspace=0.1, 
                hspace=0, 
                left=0, 
                right=1, 
                bottom=0, 
                top=1
            )
            fig.savefig(f'{path}/{step}-{i}.pdf', bbox_inches='tight')
            plt.close()
    
    @classmethod
    def from_args(cls, args, data) -> SlotAutoEncoderTrainer:
        model = SlotAutoEncoder(**args.conf.slot)
        return cls(
            model=model,
            opt=Optimizer.from_conf(model, **args.conf.optimizer),            
            device=args.device,
            dtype=args.dtype
        )
    