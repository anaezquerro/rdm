
import torch 
from itertools import product
NAND = lambda A, B: not AND(A,B)
NOR = lambda A, B: not OR(A, B)
XNOR = lambda A, B: not XOR(A,B)
IMPLIES = lambda A, B: (not A) or B
CONVERSES = lambda A, B: A or (not B)


class LogicalOperator:
    def __init__(self):
        ...
        
    def __repr__(self) -> str:
        return self.__class__.__name__
    
    def eval(self, A: torch.Tensor, B: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        return self(A, B) == S
    
    @property
    def matrix(self) -> torch.Tensor:
        combs = []
        for A, B, S in product(*[[0,1] for _ in range(3)]):
            A, B, S = map(torch.tensor, map(bool, [A, B, S]))
            if self(A,B) == S:
                combs.append(torch.stack([A, B, S]))
        return torch.stack(combs)
    
    def prob(self, p: torch.Tensor) -> torch.Tensor:
        return torch.where(self.matrix.unsqueeze(-1), p, 1-p).prod(dim=1).sum(0)

class OR(LogicalOperator):
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A | B

class AND(LogicalOperator):
    def __call__(self, A: torch.Tensor, B: torch.Tensor):
        return A & B


class XOR(LogicalOperator):
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.logical_xor(A, B)
    
    
class NAND(LogicalOperator):
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return ~(A & B)
    
class NOR(LogicalOperator):
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return ~(A | B)
    
class XNOR(LogicalOperator):
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return ~(torch.logical_xor(A, B))
    
class IMPLIES(LogicalOperator):
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return (~A) | B 
    
class CONVERSES(LogicalOperator):
    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A | (~B)