from __future__ import annotations
import torch
from typing import Callable
from torch.nn.functional import mse_loss, interpolate, avg_pool2d
from math import prod, ceil
import matplotlib.pyplot as plt 

from src.trainer import Trainer 
from src.neural import *
from src.util import Config, postprocess
from .dm import *
from .opt import Optimizer
from tqdm import tqdm 



class DiffusionTrainer(Trainer):
    """Implementation of the patch-based diffusion trainer."""
    NAME = 'base'
    
    def __init__(
        self,
        model: Denoiser, 
        opt: Optimizer,
        time: TimeSampler,
        dm: Flow | DDPM,
        device: torch.device,
        dtype: torch.dtype | str,
        unc_conf: Config = Config(weight=0.01, start_on=0, patch_size=0, force=False),
        var_conf: Config = Config(weight=0.0, start_on=0),
    ):
        super(DiffusionTrainer, self).__init__(model, opt, device, dtype)
        self.time = time.to(device)
        self.dm = dm
        self.var_conf = var_conf
        self.unc_conf = unc_conf 
        self.patch_size = unc_conf.patch_size 

        self.resolution = self.raw_model.resolution 
        self.num_channels = self.raw_model.output_dim-2
        self.img_size = self.num_channels, *self.resolution

    @property
    def grid_size(self) -> tuple[int, int]:
        """Returns the number of patches in both dimensions of an input image."""
        if self.patch_size:
            height, width = self.resolution
            return ceil(height/self.patch_size), ceil(width/self.patch_size)
        else:
            return self.resolution 
        
    @property
    def num_patches(self) -> tuple[int, int]:
        return prod(self.grid_size)

    def parse_masks(
        self,
        img: torch.Tensor,
        spatial_mask: torch.Tensor | None,
        depth_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial_mask is None:
            spatial_mask = torch.zeros_like(img, dtype=bool)[:, 0:1]
        elif spatial_mask.shape[-2:] != img.shape[-2:]:
            spatial_mask = interpolate(spatial_mask.float(), size=img.shape[-2:], mode='nearest-exact').bool()
        if depth_mask is None:
            depth_mask = torch.zeros(*img.shape[:2], dtype=bool, device=img.device)
        else:
            depth_mask = depth_mask
        return spatial_mask, depth_mask
    
    def mse_loss(
        self,
        pred: torch.Tensor,
        real: torch.Tensor,
        weight: torch.Tensor
    ) -> torch.Tensor:
        return (mse_loss(pred, real, reduction='none')*weight).mean()
        
    def unc_loss(
        self, 
        real: torch.Tensor, 
        pred: torch.Tensor, 
        spred: torch.Tensor,
        weight: torch.Tensor
    ) -> torch.Tensor:
        """Compute the uncertainty loss.
        
        Args:
            real (torch.Tensor ~ [batch_size, 1, height, width]): Real signal.
            pred (torch.Tensor ~ [batch_size, in_channels, height, width]): Predicted signal.
            spred (torch.Tensor ~ [batch_size, in_channels, height, width]): Predicted deviation.
            weight (torch.Tensor ~ [batch_size, height, width]): Time weights.

        Returns:
            torch.Tensor: KL loss.
        """
        if (self.unc_conf.start_on <= self.opt.step) and self.patch_size:
            patchify = lambda x: interpolate(avg_pool2d(x, self.patch_size), size=real.shape[-2:], mode='nearest-exact')
            spred = (0.5*patchify(spred)).exp()
            d = DiagonalGaussian(pred.detach(), spred)
            loss = d.nll(real)
            return ((loss*weight).mean()*self.unc_conf.weight).clamp(0)
        else:
            return torch.tensor(0.0, device=self.device)
        
    def mask(self, xt: torch.Tensor, img: torch.Tensor, spatial_mask: torch.Tensor, depth_mask: torch.Tensor) -> torch.Tensor:
        mask = (spatial_mask | depth_mask[..., None, None])
        return (mask*img) + (~mask*xt)

    def train_step(
        self,
        img: torch.Tensor,
        spatial_mask: torch.Tensor | None = None,
        depth_mask: torch.Tensor | None = None,
        **kwargs
    ):
        batch_size, *_ = img.shape
        spatial_mask, depth_mask = self.parse_masks(img, spatial_mask, depth_mask)
        
        # t ~ [batch_size, 1, height, width]
        # weight ~ [batch_size, 1, height, width]
        if self.unc_conf.force:
            t, weight = self.time(batch_size*self.num_patches)
            t = interpolate(t.reshape(batch_size, 1, *self.grid_size), size=self.resolution, mode='nearest')
            weight = interpolate(weight.reshape(batch_size, 1, *self.grid_size), size=self.resolution, mode='nearest')
        else:
            t, weight = self.time(batch_size)
            t = t.reshape(batch_size, 1, 1, 1).repeat(1, 1, *self.resolution)
            weight = weight.reshape(batch_size, 1, 1, 1).expand_as(t)
        xt, target = self.dm.forward(img, t)
        
        # apply masks 
        t = t*(~spatial_mask)
        weight = weight*(~spatial_mask)
        xt = self.mask(xt, img, spatial_mask, depth_mask)
        target = target[~depth_mask].view(batch_size, -1, *self.resolution)

        # the model always returns the input dimension with variance and 
        # uncertainty channels (+ optional channels)
        pred, _, spred = self.model(xt=xt, t=t, **kwargs).split([self.num_channels, 1, 1], dim=1)
        return dict(
            noise=self.mse_loss(pred, target, weight),
            unc=self.unc_loss(target, pred, spred, weight),
        )
    
    @torch.no_grad()
    def pred_step(
        self,
        img: torch.Tensor, 
        spatial_mask: torch.Tensor = None,
        depth_mask: torch.Tensor = None, 
        sampling: str = 'parallel',
        num_steps: int = 10,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        spatial_mask, depth_mask = self.parse_masks(img, spatial_mask, depth_mask)
        if sampling == 'autoregressive':
            track = self.autoregressive(img, spatial_mask=spatial_mask, depth_mask=depth_mask, **kwargs, num_steps=num_steps)
        else:            
            xt = self.mask(torch.randn_like(img), img, spatial_mask, depth_mask)
            t = torch.where(spatial_mask, 0.0, 1.0)
            track = [xt.cpu()]
            while t.sum() > 0:
                pred, v, _ = self.model(xt=xt, t=t, **kwargs).split([self.num_channels, 1, 1], dim=1)
                k = (t-1/num_steps).clamp(0.0)
                update = self.dm.backward(xt[~depth_mask].reshape(*pred.shape), t, pred, k=k, vt=v)
                xt[~depth_mask[..., None, None].expand_as(xt)] = update.flatten()
                t = k
                xt = self.mask(xt, img, spatial_mask, depth_mask)
                track.append(xt.cpu())
        track = torch.stack(track, dim=1) 
        pred = track[:, -1].to(self.device)[~depth_mask].reshape(img.shape[0], *self.img_size)
        return dict(track=track, pred=pred)
    
    @torch.no_grad()
    def autoregressive(
        self,
        img: torch.Tensor, 
        spatial_mask: torch.Tensor,
        depth_mask: torch.Tensor, 
        num_steps: int = 10,
        **kwargs
    ) -> list[torch.Tensor]:
        xt = self.mask(torch.randn_like(img), img, spatial_mask, depth_mask)
        finished = spatial_mask.clone()
        track =  [xt.cpu()]
        with tqdm(total=num_steps*self.num_patches, desc='autoregressive') as bar:
            while (~finished).any():
                t = torch.where(finished, 0.0, 1.0)
                unc = torch.zeros_like(t)
                # denoise and accumulate uncertainty
                while t.sum() > 0:
                    pred, v, u = self.model(xt=xt, t=t, **kwargs).split([self.num_channels, 1, 1], dim=1)
                    k = (t - 1/num_steps).clamp(0.0)
                    update = self.dm.backward(xt[~depth_mask].reshape(*pred.shape), t, pred, k=k, vt=v)
                    xt[~depth_mask] = update.flatten(0,1)
                    xt = self.mask(xt, img, spatial_mask, depth_mask)
                    t = k
                    unc += u
                    bar.update(1)
                    bar.set_postfix_str(f't={t.mean().item():.2f}, finished={finished.float().mean()*100:.2f}')
                # compute the lowest accumulated uncertainty 
                unc = avg_pool2d(unc, kernel_size=self.patch_size)
                unc[avg_pool2d(finished.float(), kernel_size=self.patch_size) == 1] = unc.max()+1
                prior = unc.flatten(1,-1).min(-1).values.view(unc.shape[0], 1, 1, 1)
                prior = interpolate((unc == prior).float(), size=t.shape[-2:], mode='nearest-exact').bool()
                finished |= prior
                xt = self.mask(torch.randn_like(xt), xt, finished, depth_mask)
                track.append(xt.cpu())
        return track

    def save_pred(
        self,
        step: int,
        path: str, 
        img: torch.Tensor,
        track: torch.Tensor,
        **_
    ):
        for i in range(img.shape[0]):
            num_panels = int(ceil(img.shape[1]/3))
            fig, ax = plt.subplots(1, num_panels + 5, figsize=(10, (num_panels+5)*10))

            for c, panel in enumerate(img[i].split(3, dim=0)):
                ax[c].imshow(postprocess(panel), cmap='gray')

            # plot denoised images
            for col, t in enumerate(torch.linspace(0, track.shape[1]-1, 5).int().tolist()):
                ax[col+num_panels].title.set_text(f'Step {t}')
                ax[col+num_panels].imshow(postprocess(track[i, t, :3]), cmap='gray')
            for a in ax.flatten():
                a.axis('off')
            fig.savefig(f'{path}/{step}-{i}.pdf', bbox_inches='tight')
            plt.close()
    
    @classmethod
    def from_args(cls, args, data: Callable) -> DiffusionTrainer:
        model = DenoiserUNet(**args.conf.model)
        return cls(
            model=model,
            opt=Optimizer.from_conf(model, **args.conf.optimizer),
            time=TimeSampler.from_conf(**args.conf.dm.sampler, device=args.device),
            dm=DM(**args.conf.dm),
            unc_conf=args.conf.dm.uncertainty,
            device=args.device,
            dtype=args.dtype
        )