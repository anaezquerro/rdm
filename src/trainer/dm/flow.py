import torch

class Flow:
    def __init__(self):
        ...

    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies the forward process in Rectified Flow.
        
        Args:
            x0 (torch.Tensor ~ [batch_size, in_channels, height, width]): Original sample.
            t (torch.Tensor ~ [batch_size, 1, height, width]): Continuous noise level [0,1].
        
        Returns:
            torch.Tensor ~ [batch_size, in_channels, height, width]: Latent variable.
            torch.Tensor ~ [batch_size, in_channels, height, width]: Velocity field.
        """
        e = torch.randn_like(x0)
        xt = (1 - t) * x0 + t * e
        ut = e - x0
        return xt, ut

    def backward(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor, 
        ut: torch.Tensor,
        k: torch.Tensor,
        **_
    ) -> torch.Tensor:
        """
        Applies a backward step in ODE.

        Args:  
            xt (torch.Tensor ~ [batch_size, in_channels, height, width]): Noisy signal.
            t (torch.Tensor ~ [batch_size, 1, height, width]): Current timestep between [0, 1].
            k (torch.Tensor ~ [batch_size, 1, height, width]): Previous timestep between [0, 1].
            ut (torch.Tensor ~ [batch_size, in_channels, height, width]): Velocity field.
        """
        dt = k - t
        pred = torch.where(t > 0, xt + dt * ut, xt)
        return pred 