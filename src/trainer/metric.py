from __future__ import annotations
import torch 
from torch.nn.functional import mse_loss
from torch import nn 
from src.util import sync

class Metric:
    ATTRIBUTES: list[str]
    METRICS: list[str]
    BETTER: list[int]
    
    def __init__(self, model: nn.Module = None):
        if model is not None:
            model = model.requires_grad_(False)
            model.eval()
        self.model = model 
            
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, torch.tensor(0.0))
            
    def __repr__(self) -> str:
        return ', '.join(f'{metric}={self.__getattribute__(metric):.2f}' for metric in self.METRICS)
    
    def to(self, device: torch.device) -> Metric:
        for name in self.ATTRIBUTES:
            self.__setattr__(name, self.__getattribute__(name).to(device))
        if self.model is not None:
            self.model = self.model.to(device)
        return self
    
    def display(self, step: int, path: str, **_):
        pass
    
    def sync(self) -> Metric:
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, sync(self.__getattribute__(attr)))
        return self
    
    def improves(self, other: Metric) -> bool:
        for metric, better in zip(self.METRICS, self.BETTER):
            if self.__getattribute__(metric)*better > getattr(other, metric)*better:
                return True 
        return False 
    
    def copy(self) -> Metric:
        new = self.__class__(self.model)
        for attr in self.ATTRIBUTES:
            setattr(new, attr, self.__getattribute__(attr))
        return new
    
    def __add__(self, other: Metric) -> Metric:
        new = self.__class__(self.model)
        for attr in self.ATTRIBUTES:
            setattr(new, attr, self.__getattribute__(attr) + getattr(other, attr))
        return new 
    
    def __radd__(self, other: Metric) -> Metric:
        return self + other 
    
    def __call__(self, *args, **kwargs) ->  dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def reset(self, device: torch.device) -> Metric:
        new = self.copy()
        for attr in self.ATTRIBUTES:
            setattr(new, attr, torch.tensor(0.0))
        return new.to(device)
    
    def save(self, path: str):
        torch.save(self, path)
        
    @classmethod
    def load(cls, path: str, device: torch.device) -> Metric:
        return torch.load(path, map_location=device, weights_only=False)
    


class ReconstructionMetric(Metric):
    ATTRIBUTES = ['mse', 'total']
    METRICS = ['MSE']
    BETTER = [1]

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, **_) -> ReconstructionMetric:
        self.mse += mse_loss(pred, target)
        self.total += 1
        return self 
    
    @property
    def MSE(self) -> float:
        return float(self.mse/self.total)
    
class AccuracyMetric(Metric):
    ATTRIBUTES = ['correct', 'total']
    METRICS = ['ACC']
    BETTER = [1]
    
    def __call__(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,  
        **_
    ) -> AccuracyMetric:
        """Accuracy metric update.

        Args:
            pred (torch.Tensor ~ [batch_size, num_targets]): Prediction.
            target (torch.Tensor ~ [batch_size, num_targets]): Targets.
            
        Returns:
            Metric update.
        """
        self.correct += (pred == target).sum().detach()
        self.total += target.numel()
        return self

    @property
    def ACC(self) -> float:
        return self.correct/self.total*100

