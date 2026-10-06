from __future__ import annotations
from .tangram import Tangram 
import torch, random 
import numpy as np 
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, RandomVerticalFlip
from torch.nn.functional import one_hot

from src.trainer.dataset import Dataset
from src.env import DATA_FOLDER
from .metric import TangramMetric

class TangramPreprocess:
    def __init__(self, max_retries: int = 10):
        self.transform = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5)
        ])
        self.max_retries = max_retries

    def __call__(self, tangram: torch.Tensor, piece: torch.Tensor):
        transformed = self.transform(tangram.unsqueeze(0)).squeeze(0)
        indices = transformed.int().argmax(0)
        transformed = one_hot(indices, num_classes=transformed.shape[0]).movedim(-1,0).bool() & transformed.any(0)
        piece = self.transform(piece.unsqueeze(0)).squeeze(0)
        return transformed, piece 

class TangramDataset(Dataset):
    """Implementation of the Tangram dataset, with four difficulty versions.
    
    - Static: Simple partition of the sihlouette with static colors.
    - Canon: Partition of the sihlouette with random coloring of the canon Tangram (the square).
    - Full: Partition of the sihlouette with moving shuffles.
    """
    COLORS: torch.Tensor = torch.tensor([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [102, 51, 153],
        [255, 140, 0],
        [0, 128, 128]  
    ], dtype=torch.uint8)
    NUM_CHANNELS: dict[str, int] = dict(solution=3, piece=3, layout=1)
    SAMPLE = Tangram
    RESOLUTION: tuple[int, int] = (128, 128)

    def __init__(self, piece: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.piece = piece 
        self.preprocess = TangramPreprocess()

    def RS(self, i: int) -> dict[str, torch.Tensor]:
        tangram = self.data[i]
        tangram, piece = self.preprocess(tangram, random.choice(self.piece))

        # prepare sihlouette layout
        layout = (tangram.any(0)*255).to(torch.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # layout = torch.from_numpy(cv2.morphologyEx(layout.numpy(), cv2.MORPH_CLOSE, kernel)).squeeze(0)

        # color tangram and pieces
        colored, piece, color = self.color(tangram, piece)

        sample = dict(
            solution=colored,
            piece=piece,
            layout=layout.unsqueeze(0),
            depth_mask=torch.tensor([False, False, False, True, True, True, True], dtype=bool),
            tangram=tangram,
            color=color
        )
        return sample 
    
    def OL(self, i: int) -> dict[str, torch.Tensor]:
        sample = self.RS(i)
        sample['img'] = sample['piece' if random.uniform(0,1) > 0.3 else 'solution']
        return sample

    def color(self, tangram: torch.Tensor, piece: torch.Tensor):
        """Color tangram and piece views.
        
        Args:
            tangram (torch.Tensor ~ [7, *size]): Tangram decomposition.
            piece (torch.Tensor ~ [7, *size]): Piece decomposition.

        Returns: 
            tuple[torch.Tensor ~ [3, 128, 128]]: RGB representation of Tangram and pieces.
        """
        index = torch.randperm(len(self.COLORS))[:7]
        color = self.COLORS[index].view(7, 3, 1, 1)
        tangram = (color*tangram.unsqueeze(1).int()).sum(0)
        piece = (color * piece.unsqueeze(1)).int().sum(0)
        return tangram.to(torch.uint8), piece.to(torch.uint8), index

    @property
    def METRIC(self) -> TangramMetric:
        return TangramMetric()
    
    @classmethod
    def from_folder(cls, split: str, **_):
        # data ~ [num_samples, 7, *size]
        data = torch.from_numpy(np.unpackbits(np.load(f'{DATA_FOLDER}/tangram/{split}.npy')).reshape(-1, 7, *cls.RESOLUTION).astype(bool))
        piece = torch.from_numpy(np.unpackbits(np.load(f'{DATA_FOLDER}/tangram/shuffles.npy')).reshape(-1, 7, *cls.RESOLUTION).astype(bool))
        return cls(data=data, piece=piece, **_)
