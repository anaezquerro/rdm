from torch import nn 

def Activation(name: str) -> nn.Module:
    match name: 
        case 'silu':
            return nn.SiLU()
        case 'relu':
            return nn.ReLU()
        case 'leakyrelu':
            return nn.LeakyReLU(0.1)
        case 'linear':
            return nn.Identity()
        case 'gelu':
            return nn.GELU()
        case _:
            raise ValueError(f'Activation {name} not available')