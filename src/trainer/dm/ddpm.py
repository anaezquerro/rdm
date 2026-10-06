from __future__ import annotations
import torch 
from .var import VarianceScheduler

class DDPM:
    def __init__(self, scheduler: VarianceScheduler):
        self.scheduler = scheduler
        
    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Directly sample x(t) from x(0).
        
        Args:  
            x0 (torch.Tensor ~ [batch_size, in_channels, height, width]): Input feature map.
            t (torch.Tensor ~ [batch_size, 1, height, width]): Continuous timestep [0,1].
            
        Returns:
            torch.Tensor ~ [batch_size, in_channels, height, width]: Latent variable.
            torch.Tensor ~ [batch_size, in_channels, height, width]: Gaussian noise.
        """
        at = self.scheduler.a(t)
        mt = self.mean(x0, t)
        e = torch.where(t > 0, torch.randn_like(x0), 0.0)
        return mt + (1-at).sqrt()*e, e
    
    def mean(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        at = self.scheduler.a(t)
        return at.sqrt()*x0
    
    def mean_pred(self, xt: torch.Tensor, t: torch.Tensor, et: torch.Tensor):
        at =  self.scheduler.a(t)
        return (xt - (1-at).sqrt()*et)/at.sqrt()
    
    def backward(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor, 
        k: torch.Tensor,
        et: torch.Tensor, 
        vt: torch.Tensor = None,
        **_
    ) -> torch.Tensor:
        """Applies one backward step in DDPM.
        
        Args:  
            xt (torch.Tensor ~ [batch_size, in_channels, height, width]): Noisy signal.
            t (torch.Tensor ~ [batch_size, 1, height, width]): Continuous timestep.
            k (torch.Tensor ~ [batch_size, 1, height, width]): Previous timestep.
            et (torch.Tensor ~ [batch_size, in_channels, height, width]): Predicted noise.
            vt (torch.Tensor ~ [batch_size, in_channels, height, width]): Predicted variance.
            
        Returns:
            torch.Tensor ~ [batch_size, in_channels, height, width]: Previous latent.
        """
        at  = self.scheduler.a(t)
        ak = self.scheduler.a(k)

        # estimate x0 
        x0 = (xt - (1-at).sqrt()*et)/at.sqrt().clamp(1e-8)

        # posterior mean coefficients
        ct = (at / ak).sqrt() * (1 - ak) / (1 - at).clamp(1e-8)
        c0 = (ak.sqrt() * (1 - at / ak)) / (1 - at).clamp(1e-8)
        mk = c0 * x0 + ct * xt
        
        # compute variance 
        if vt is None:
            st = self.scheduler.std(t)
        else:
            st = vt.sqrt()
        e = torch.where(k > 0, st*torch.randn_like(xt), 0.0)
        return torch.where(t > 0,  mk + e, xt)
    

class DDIM(DDPM):

    def backward(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor, 
        k: torch.Tensor,
        et: torch.Tensor,
        **_
    ):
        at  = self.scheduler.a(t)
        ak = self.scheduler.a(k)
        x0 = (xt - (1-at).sqrt()*et)/at.sqrt().clamp(1e-8)
        xk = ak.sqrt()*x0 + (1-ak).sqrt()*et
        return torch.where(t > 0,  xk, xt)
    