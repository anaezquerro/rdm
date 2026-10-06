from __future__ import annotations
from torch.optim import AdamW, Adam, RMSprop
from torch import nn
import torch 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler, LinearLR, ConstantLR

from src.util import Config, WORLD_SIZE, sync, avg
from .scheduler import SlotAttentionScheduler

class Optimizer:
    """Custom class to configure the optimizer steps."""

    def __init__(
        self,
        opt: torch.optim.Optimizer,
        scheduler: LRScheduler,
        batch_size: int,
        max_norm: float,
        ema: EMA | None
    ):
        """Optimizer initialiation.

        Args:
            opt (Optimizer): PyTorch optimizer.
            scheduler (LRScheduler): Learning rate scheduler.
            batch_size (int): Effective batch size. Steps will be computed dynamically.
            max_norm (float): Clip gradient norm.
        """
        self.opt = opt 
        self.scheduler = scheduler
        self.step = 0
        self.max_norm = max_norm
        self.ema = ema 
        self.track = []
        self.batch_size = batch_size 
        self.ac_samples = 0

    def __repr__(self) -> str:
        return f'Optimizer(\n'+ \
            f'  scheduler={self.scheduler}, batch_size={self.batch_size}, max_norm={self.max_norm},\n' + \
            f'  ema={self.ema}\n)'
    
    def __call__(
        self, 
        loss: torch.Tensor, 
        model: nn.Module, 
        num_samples: int
    ) -> torch.Tensor:
        """Performs the backward computation and weight update.

        Args:
            loss (torch.Tensor): Model loss.
            model (nn.Module): Model.
            num_samples (int): Number of samples from which the loss has been computed.
        """
        self.ac_samples += (num_samples * WORLD_SIZE)
        scaled_loss = loss*(num_samples/self.batch_size)
        scaled_loss.backward()
        if self.ac_samples >= self.batch_size:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_norm)
            self.opt.step()
            self.scheduler.step()
            self.opt.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(model)
            self.ac_samples = 0
            self.step += 1
            self.track.append(sync(loss, op=dist.ReduceOp.AVG).detach().cpu().item())
        return loss 

    @property
    def status(self) -> dict[str, float]:
        return dict(lr=self.lr, ac=self.ac_samples)
    
    @property
    def lr(self) -> float:
        return self.opt.param_groups[0]['lr']

    def slope(self, steps: int) -> float:
        steps = min(steps, len(self.track)//2)
        x1 = self.track[-(steps*2):-steps]
        x2 = self.track[-steps:]
        return avg(x2)-avg(x1)
    
    def loss(self, steps: int) -> float:
        return sum(self.track[-steps:])/min(steps, len(self.track))

    @property
    def state(self):
        state = dict(
            optimizer_state=self.opt.state_dict(), 
            step=self.step,
            track=self.track,
            scheduler_state=self.scheduler.state_dict()
        )
        if self.ema is not None:
            state['ema_params'] = self.ema.shadow_params
        return state
    
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> Optimizer:
        self.opt.load_state_dict(state['optimizer_state'])
        self.scheduler.load_state_dict(state['scheduler_state'])
        self.track = state.get('track', [])
        self.step = state.get('step', 0)
        if 'ema_params' in state and self.ema is not None:
            self.ema.shadow_params = state['ema_params']
        return self 
    
    @classmethod
    def from_conf(
        cls, 
        model: nn.Module, 
        name: str,
        lr: float,
        batch_size: int,
        max_norm: float,
        ema: float,
        scheduler: Config = Config(name='constant'),
        **kwargs
    ) -> Optimizer:
        opt = cls.get_optimizer(model, name=name, lr=lr, **kwargs)
        sched = cls.get_scheduler(opt, **scheduler)
        ema = EMA(model, ema) if ema else None
        return cls(opt, sched, batch_size=batch_size, max_norm=max_norm, ema=ema)
    
    @staticmethod
    def get_optimizer(model: nn.Module, name: str, **kwargs) -> torch.optim.Optimizer:
        match name:
            case 'adamw':
                return AdamW(model.parameters(), **kwargs)
            case 'adam':
                return Adam(model.parameters(), **kwargs)
            case 'rmsprop':
                return RMSprop(model.parameters(), **kwargs)
            case _:
                raise ValueError(f'Optimizer {name} not available')
            
    @staticmethod
    def get_scheduler(opt: torch.optim.Optimizer, name: str, **kwargs) -> LRScheduler:
        match name:
            case 'linear':
                return LinearLR(opt, **kwargs)
            case 'constant':
                return ConstantLR(opt, factor=1, total_iters=1)
            case 'slot':
                return SlotAttentionScheduler(opt, **kwargs)
            case _:
                raise ValueError(f'Scheduler {name} not available')

class EMA:
    def __init__(self, model: nn.Module | DDP, decay: float):
        # The main, active model whose methods we want to use
        model = model if not isinstance(model, DDP) else model.module
        self.decay = decay
        
        # copy model weights for initialization
        self.shadow_params = {name: param.clone().detach().cuda() for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module | DDP):
        """Update the shadow weights using the EMA rule."""
        model = model if not isinstance(model, DDP) else model.module
        for name, param in model.named_parameters():
            shadow_param = self.shadow_params[name]
            new_average = self.decay * shadow_param + (1. - self.decay) * param
            shadow_param.copy_(new_average.data)

    def broadcast(self, model: nn.Module):
        """Broadcasts shadow parameters to the model."""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow_params[name].data)
            
    def __repr__(self) -> str:
        return f'EMA(decay={self.decay})'

