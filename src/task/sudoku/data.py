from __future__ import annotations
import torch, random
from kornia.geometry.transform import resize
import numpy as np

from src.env import RESULTS_FOLDER, DATA_FOLDER, EVAL_FOLDER
from .eval import MNISTDataset
from src.util import shuffle
from src.trainer import Dataset
from .metric import SudokuMetric, SudokuReconstructionMetric
from src.trainer.transform import RandomAlignedPerspective

class SudokuDataset(Dataset):
    """Implementation of the Sudoku dataset, with combinatorial variants (colored and tilted)."""
    NUM_CHANNELS: dict[str, int] = dict(solution=3, clue=3, img=3)
    COLORS = MNISTDataset.COLORS

    def __init__(
        self,
        mnist: list[torch.Tensor],
        colored: bool = False,
        tilted: float = 0.0,
        level: tuple[int, int] = (1, 80),
        **_
    ):
        super().__init__(**_)
        self.mnist = mnist
        self.colored = colored 
        self.tilted = tilted 
        self.level = level 
        self.preprocess = RandomAlignedPerspective(self.tilted, interp='exact' if colored else 'bilinear')

    def PB(self, i: int) -> dict[str, torch.Tensor]:
        label = random.uniform(0, 1) < 0.5 
        if label:
            solution = self.RS(i)['solution']
        else: # prepare false sudoku
            digit, color = self.data[i % len(self.data)].unbind(-1)
            # digit = digit.flatten()[torch.randperm(digit.numel())].reshape(digit.shape)
            color = torch.tensor(shuffle(range(1,10))).repeat(9).reshape(9,9)
            solution = self.img(digit)
            if self.colored:
                solution = self.color(solution, color)
            else:
                solution = solution.expand(3, -1, -1) # set to RGB for consistency
                color = torch.zeros_like(color)
        return dict(solution=solution, label=label)

    def RS(self, i: int) -> dict[str, torch.Tensor]:
        digit, color = self.data[i % len(self.data)].unbind(-1)
        solution = self.img(digit)
        mask = self.mask()
        spatial_mask = resize(mask.float().unsqueeze(0), size=solution.shape[-2:], interpolation='nearest') > 0.5
        if self.colored:
            solution = self.color(solution, color)
        else:
            color = torch.zeros_like(color)
        clue = torch.where(spatial_mask, solution, 0).to(torch.uint8) # empty cells are set to black
        sample = dict(solution=solution, clue=clue, spatial_mask=spatial_mask, digit=digit, color=color, mask=mask)
        if self.tilted:
            self.tilt(sample)
        return sample
    
    def OL(self, i: int) -> dict[str, torch.Tensor]:
        batch = self.RS(i)
        return batch

    def mask(self) -> torch.Tensor:
        indices = shuffle(range(81))[:random.randint(*self.level)]
        mask = torch.zeros(9, 9, dtype=bool)
        for index in indices:
            row, col = index//9, index%9
            mask[row, col] = True 
        return mask

    def sample_mnist(self, num: int) -> torch.Tensor:
        """Sample MNIST images.
        
        Returns:
            torch.Tensor[uint8] ~ [10, 9, 1, 28, 28]
        """
        pool = [img[torch.randperm(img.shape[0])[:num]] for img in self.mnist]
        return torch.stack(pool)

    def img(self, digit: torch.Tensor) -> torch.Tensor:
        """Creates the MNIST Sudoku.
        
        Args:
            digit (torch.Tensor[int] ~ [9, 9]): Sudoku grid.

        Returns:
            torch.Tensor[uint8] ~ [1, 252, 252]: MNIST Sudoku image.
        """
        pool = self.sample_mnist(9)
        img = pool[digit, range(9)]
        pool = img.permute(2, 0, 3, 1, 4).reshape(1, 9*28, 9*28)
        return pool.to(torch.uint8)
    
    def color(self, img: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
        """Apply color to a Sudoku image.
        
        Args:
            img (torch.Tensor[uint8] ~ [1, height, width]): Sudoku image.
            color (torch.Tensor[int] ~ [9, 9]): Colors to apply.

        Returns:
            torch.Tensor[uint8] ~ [3, height, width]: Colored Sudoku.
        """
        palette = self.COLORS[color].permute(2,0,1).float()
        palette = resize(palette, size=img.shape[-2:], interpolation='nearest')
        return torch.where(img > 100, palette, 0).to(torch.uint8)
    
    def tilt(self, sample: dict[str, torch.Tensor]):
        (img, clue, spatial_mask), meta = self.preprocess(sample['solution'], sample['clue'], sample['spatial_mask'].float())
        sample['solution'] = img 
        sample['clue'] = clue
        sample['spatial_mask'] = spatial_mask.bool()
        sample['meta'] = meta 
     
    @property
    def METRIC(self) -> SudokuMetric:
        post = self.preprocess.reverse if self.tilted else None
        match self.mode:
            case 'RS':
                return SudokuMetric.from_model(f'{RESULTS_FOLDER}/mnist/last.pt', post)
            case 'OL':
                return SudokuReconstructionMetric.from_model(f'{RESULTS_FOLDER}/mnist/last.pt', post)
    
    @classmethod
    def from_folder(cls, split: str, **_) -> SudokuDataset:
        mnist = torch.load(f'{EVAL_FOLDER}/mnist/mnist.pt', weights_only=False, map_location='cpu')

        # organize a dictionary
        img, digit = mnist['img'], mnist['digit']
        mnist: list[torch.Tensor] = [img[digit == i] for i in range(10)]

        data = np.load(f'{DATA_FOLDER}/mnist_sudokus.npy')
        if split == 'train':
            digit = data[:-5000]
        else:
            digit = data[-5000:]
        digit = torch.from_numpy(digit)
        color = digit[torch.randperm(digit.shape[0])]
        data = torch.stack([digit, color], dim=-1)
        return cls(data=data, mnist=mnist, **_)


        

