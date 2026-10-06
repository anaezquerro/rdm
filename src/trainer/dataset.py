from __future__ import annotations
import torch

from .transform import Transform
from src.env import DETERMINISTIC, SEED
from src.util import set_seed

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data: torch.Tensor,
        mode: str = 'RS', 
        transform: list[Transform] = [],
        training: bool = True,
        **_
    ):
        self.data = data 
        self.mode = mode 
        self.transform = transform 
        self.training = training 
        self.n = len(data)

    def __len__(self) -> int:
        return self.n 

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n={len(self)})'
    
    def __getitem__(self, i: int) -> dict[str, torch.Tensor | int]:
        if DETERMINISTIC:
            set_seed(SEED + i)
        sample = getattr(self, self.mode)(i % len(self.data))
        for transform in self.transform:
            sample[transform.name] = transform(**sample)
        return sample 
