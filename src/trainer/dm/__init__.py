from .sampler import TimeSampler, Beta
from .dgaus import DiagonalGaussian
from .ddpm import DDPM, DDIM
from .flow import Flow
from .var import VarianceScheduler

from src.util import Config 

def DM(
    name: str, 
    variance: Config | None = None,
    **_
) -> DDPM | DDIM | Flow:
    if name == 'ddpm':
        return DDPM(VarianceScheduler.from_conf(**variance))
    elif name == 'ddim':
        return DDIM(VarianceScheduler.from_conf(**variance))
    elif name == 'flow':
        return Flow()
    else:
        raise ValueError(f"Unknown diffusion model: {name}")