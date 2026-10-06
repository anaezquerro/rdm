import torch 

def normal_kl(
    mean1: torch.Tensor, 
    logvar1: torch.Tensor,
    mean2: torch.Tensor, 
    logvar2: torch.Tensor
) -> torch.Tensor:
    """
    Compute the KL divergence between two gaussians.
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + (logvar1 - logvar2).exp()
        + ((mean1 - mean2) ** 2) * (-logvar2).exp()
    )

