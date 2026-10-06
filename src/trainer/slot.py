from __future__ import annotations
from typing import Callable

from .diffusion import DiffusionTrainer
from .opt import Optimizer
from .dm import *
from src.neural import * 

class AbstractorDiffusionTrainer(DiffusionTrainer):
    NAME = 'abs'

    @classmethod
    def from_args(cls, args, data: Callable):
        assert any(args.conf.model.unet.cross_attn or args.conf.model.unet.shared_attn), 'Warning: At least one value of model cross-attention needs to be activated for conditioninig'
        args.conf.model.rel.inputs = args.conf.data.transform.slot.source
        model = AbsUNet(**args.conf.model)
        return cls(
            model=model,
            opt=Optimizer.from_conf(model, **args.conf.optimizer),
            time=TimeSampler.from_conf(**args.conf.dm.sampler, device=args.device),
            dm=DM(**args.conf.dm),
            unc_conf=args.conf.dm.uncertainty,
            device=args.device,
            dtype=args.dtype
        )


class FeatureDiffusionTrainer(AbstractorDiffusionTrainer):
    NAME = 'feat'

    @classmethod
    def from_args(cls, args, data: Callable):
        assert any(args.conf.model.unet.cross_attn), 'Warning: At least one value of model cross-attention needs to be activated for conditioninig'
        model = FeatureUNet(**args.conf.model)
        return cls(
            model=model,
            opt=Optimizer.from_conf(model, **args.conf.optimizer),
            time=TimeSampler.from_conf(**args.conf.dm.sampler, device=args.device),
            dm=DM(**args.conf.dm),
            unc_conf=args.conf.dm.uncertainty,
            device=args.device,
            dtype=args.dtype
        )
        