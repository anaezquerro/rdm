
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

def SlotAttentionScheduler(optimizer: Optimizer, warmup: int, decay: int, rate: float):
    """Implementation of the Slot Attention learning rate scheduler.
    
    Args:
        optimizer (Optimizer): PyTorch optimizer.
        warmup (int): Warmup steps.
        decay (int): Number of decay steps.
        rate (float): Decay rate.

    Returns:
        LambdaLR
    """
    def scheduler(step: int) -> float:
        step = max(1, step)
        if step < warmup:
            return step/max(1, warmup)
        return rate ** ((step - warmup)/decay)
    return LambdaLR(optimizer, scheduler)
        