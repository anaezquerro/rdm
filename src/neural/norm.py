from torch import nn

def Norm(name, dim: int) -> nn.Module:
    match name:
        case 'batch':
            norm = nn.BatchNorm2d(dim)
        case 'group':
            norm = nn.GroupNorm(dim//4, dim)
        case 'instance':
            norm = nn.InstanceNorm2d(dim, affine=True)
        case 'layer':
            norm = nn.LayerNorm(dim)
        case _:
            return nn.Identity()
    nn.init.ones_(norm.weight)
    nn.init.zeros_(norm.bias)
    return norm
            

    
