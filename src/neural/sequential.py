
import torch
import torch.nn as nn

class Sequential(nn.Sequential):
    def forward(self, *args):
        for module in self:
            args = module(*args)
            if isinstance(args, torch.Tensor):
                args = (args,)
        if len(args) == 1:
            return args[0]
        return args 
    

class Identity(nn.Identity):
    def forward(self, x, *args, **kwargs):
        return x
