
from __future__ import annotations
import torch


class VarianceScheduler:
    """Abstract implementation of a variance scheduler.
    Note that b(t) is going to denote the instantaneous noise rate,
    while a(t) the accumulated variance term for direct sampling.

    The code assumes continuous noise (in the range [0, 1]).
    """
    def b(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def a(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def var(self, t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > 0, 1-self.a(t), 0.0)

    def std(self, t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > 0, self.var(t).sqrt(), 0.0)

    @classmethod
    def from_conf(cls, name: str, **kwargs) -> VarianceScheduler:
        if name == 'linear':
            return LinearScheduler(**kwargs)
        elif name == 'cosine':
            return CosineScheduler(**kwargs)
        else:
            raise ValueError(f"Unknown variance scheduler: {name}")
    
class LinearScheduler(VarianceScheduler):
    """Implementation of the linear scheduler for continuous noise."""
    def __init__(self, beta1: float = 0.1, betaT: float = 2.0):
        self.beta1 = beta1 
        self.betaT = betaT 

    def b(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta1 + (self.betaT - self.beta1) * t

    def a(self, t: torch.Tensor) -> torch.Tensor:
        integral = self.beta1 * t + 0.5 * (self.betaT - self.beta1) * (t**2)
        return (-integral).exp()
    

class CosineScheduler(VarianceScheduler):
    """Implementation of the cosine scheduler for continuous noise."""
    def __init__(self, s: float = 0.008, vmax: int = 2.0):
        self.s = s 
        self.f0 = torch.cos(torch.tensor(s / (1 + s) * torch.pi / 2))**2
        self.vmax = vmax

    def b(self, t: torch.Tensor) -> torch.Tensor:
        phi = ((t + self.s) / (1 + self.s)) * (torch.pi / 2)
        return ((torch.pi / (1 + self.s)) * torch.tan(phi)).clamp(0, self.vmax)

    def a(self, t: torch.Tensor) -> torch.Tensor:
        phi = ((t + self.s) / (1 + self.s)) * (torch.pi / 2)
        ft = torch.cos(phi)**2
        return (ft / self.f0).clamp(1e-5, 1.0)