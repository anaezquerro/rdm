from .trainer import Trainer 
from .ae import AutoEncoderTrainer, SlotAutoEncoderTrainer
from .metric import Metric, ReconstructionMetric, AccuracyMetric
from .dataset import Dataset
from .transform import Transform
from .diffusion import DiffusionTrainer
from .slot import AbstractorDiffusionTrainer, FeatureDiffusionTrainer
from .dm import *
from .opt import Optimizer, EMA

import inspect

TRAINERS = {
    obj.NAME: obj 
    for name, obj in globals().items() 
    if inspect.isclass(obj) and hasattr(obj, "NAME") and name.endswith("Trainer")
}


