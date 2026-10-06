from __future__ import annotations
import torch, random, pickle 
import pandas as pd 
from src.env import DATA_FOLDER, RESULTS_FOLDER, EVAL_FOLDER
from .operator import XOR, OR, AND, IMPLIES, LogicalOperator
from .metric import LogicFaceMetric, LogicFaceReconstructionMetric
from .eval import CelebADataset

from torchvision.transforms.v2.functional import to_pil_image
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip
import matplotlib.pyplot as plt
    
class LogicFaceDataset(CelebADataset):
    NUM_CHANNELS: int = dict(A=3, B=3, S=3, img=3, clue=6)
    OPERATORS: list[LogicalOperator] = [OR(), AND(), XOR(), IMPLIES()]
    RANDOMIZE: Compose = Compose([
        RandomHorizontalFlip(p=0.5)
    ])

    def __init__(self, panels: dict[str, tuple[torch.Tensor, torch.Tensor]], **kwargs):
        super(CelebADataset, self).__init__(**kwargs)
        self.panels = panels 
        self.names = list(panels.keys())

    def OL(self, i: int) -> dict[str, torch.Tensor]:
        img, feat = self.panels[random.choice(self.names)]
        return dict(img=img, attr=feat)

    def RS(self, i: int) -> dict[str, torch.Tensor]:
        """Retrieve a sample for the reasoning task.
        
        Args:
            i (int): Sample index.
            
        Returns:
            A, B, S (list[PIL.Image]): Face images for the A, B and S logical arguments.
            mask (torch.Tensor ~ 3): Boolean tensor to mask one of the arguments (A, B, S).
            ops (torch.Tensor ~ [num_feats]): Logical operator used for each feature.
            feats (torch.Tensor ~ [3, num_feats]): Feature vector of each image.
        """
        A, B, S = map(self.panels.get, self.data[i])
        feat = torch.stack([A[1], B[1], S[1]])
        A, B, S = map(self.RANDOMIZE, (A[0], B[0], S[0]))
        mask = torch.tensor([True, True, False]).repeat_interleave(3)
        return dict(A=A, B=B, S=S, 
                    depth_mask=mask, 
                    attr=feat, names=self.data[i])
    
    def display(self, i: int) -> plt.Figure:
        sample = self.RS(self.samples[i])
        A, B, S = map(to_pil_image, (sample['A'], sample['B'], sample['S']))

        fig, ax = plt.subplots(1, 3, figsize=(5*3, 5))
        ax[0].imshow(A)
        ax[0].title.set_text(sample['feats'][0].int().tolist())
        ax[0].set_xlabel(sample['names'][0])
        ax[1].imshow(B)
        ax[1].title.set_text(sample['feats'][1].int().tolist())
        ax[1].set_xlabel(sample['names'][1])
        ax[2].imshow(S)
        ax[2].title.set_text(sample['feats'][2].int().tolist())
        ax[2].set_xlabel(sample['names'][2])
        fig.suptitle(f'features: {self.FEATS}')
        fig.show()
    
    @property
    def METRIC(self) -> LogicFaceMetric:
        if self.mode == 'RS':
            metric = LogicFaceMetric.from_model(f'{RESULTS_FOLDER}/celeba/last.pt')
            metric.METRICS += self.FEATS
        else:
            metric = LogicFaceReconstructionMetric.from_model(f'{RESULTS_FOLDER}/celeba/last.pt')
        metric.FEATS = self.FEATS
        metric.OPERATORS = self.OPERATORS
        return metric 
    
    @classmethod
    def from_folder(cls, split: str = 'train', **_) -> LogicFaceDataset:
        with open(f'{DATA_FOLDER}/logicface/{split}.pkl', 'rb') as reader:
            cache = pickle.load(reader)
        annotations = pd.read_csv(f'{EVAL_FOLDER}/celeba/annotations.csv', index_col=0) > 0
        data = torch.load(f'{DATA_FOLDER}/logicface/imgs.pt', weights_only=False, map_location='cpu')
        panels = {path: (img, torch.tensor(feat.tolist())) for (path, feat), img in zip(annotations.iterrows(), data.unbind(0))}
        return cls(data=cache, panels=panels, **_)
