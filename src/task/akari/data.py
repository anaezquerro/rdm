from __future__ import annotations
import torch, random
from kornia.geometry.transform import resize
from PIL import Image, ImageFont, ImageDraw
from torchvision.transforms.v2.functional import pil_to_tensor
from itertools import product
import numpy as np 

from src.trainer import Dataset 
from .metric import AkariMetric, AkariReconstructionMetric
from .akari import Akari 
from src.env import DATA_FOLDER, FONT



def draw_digit(digit: int, size: int) -> torch.Tensor:
    font = ImageFont.truetype(FONT, size=int(size*0.7))
    img = Image.new('RGB', size=(size, size), color='black')
    draw = ImageDraw.Draw(img)
    draw.text((size//3, size*0.1), str(digit), font=font, fill='white', align='center')
    return pil_to_tensor(img)


def get_bulb(size: int) -> torch.Tensor:
    BULB = Image.new('RGBA', (size, size), color='#FFF59D')
    icon_size = int(size*0.8)
    ICON = Image.open(f'datasets/bulb.png').convert("RGBA").resize(
            (icon_size, icon_size), Image.Resampling.LANCZOS
    )
    offset = (size - icon_size)//2
    BULB.paste(ICON, (offset, offset), ICON)
    return pil_to_tensor(BULB.convert('RGB'))

class AkariDataset(Dataset):
    SAMPLE = Akari
    NUM_CHANNELS: dict[str, int] = dict(solution=3, clue=3, img=3)
    CELL_SIZE: int = 18

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.BULB = get_bulb(self.CELL_SIZE)
        # pre-load images as PyTorch tensors
        self.LIGHT = torch.tensor([255, 245, 157], dtype=torch.uint8)
        self.DIGIT = []
        for digit in range(5):
            self.DIGIT.append(draw_digit(digit, self.CELL_SIZE))
    
    def OL(self, i: int) -> dict[str, torch.Tensor]:
        sample = self.RS(i)
        if random.uniform(0, 1) < 0.5:
            sample['img'] = sample['solution']
            sample['spatial_mask'] = torch.ones_like(sample['spatial_mask'])
        else:
            sample['img'] = sample['clue']
        return sample
        
    def RS(self, i: int) -> dict[str, torch.Tensor]:
        wall, bulb, digit = self.data[i].split([1, 1, 5], dim=-1)
        wall, bulb = wall.squeeze(-1), bulb.squeeze(-1)
        solution = self.render(wall, bulb, digit)
        spatial_mask = resize(wall.float(), size=solution.shape[-2:], interpolation='nearest').bool().unsqueeze(0)
        clue = torch.where(spatial_mask, solution, 255).to(torch.uint8)
        return dict(solution=solution, clue=clue, spatial_mask=spatial_mask, wall=wall, bulb=bulb, digit=digit)

    def render(self, wall: torch.Tensor, bulb: torch.Tensor, digit: torch.Tensor) -> torch.Tensor:
        H, W = Akari.GRID_SIZE
        C = self.CELL_SIZE
        img = torch.full((3, H*C, W*C), 255, dtype=torch.uint8)
        img[:] = self.LIGHT.view(-1, 1, 1)

        # add walls and digits 
        for r, c in product(range(H), range(W)):
            x0, y0 = r * C, c * C
            x1, y1 = x0 + C, y0 + C
                
            if wall[r, c]:
                img[:, x0:x1, y0:y1] = 0
                if digit[r, c].any():
                    img[:, x0:x1, y0:y1] += self.DIGIT[digit[r,c].nonzero().item()]

        # add bulbs only on the image
        for r, c in bulb.nonzero().tolist():
            x0, y0 = r * C, c * C
            x1, y1 = x0 + C, y0 + C
            img[:, x0:x1, y0:y1] = self.BULB
        return img.contiguous()
    
    @property
    def METRIC(self) -> AkariReconstructionMetric:
        args = dict(LIGHT=self.LIGHT, BULB=self.BULB, DIGIT=self.DIGIT)
        if self.mode == 'OL':
            return AkariReconstructionMetric(**args)
        else:
            return AkariMetric(**args)

    @classmethod
    def from_folder(cls, split: str, **_):
        data = torch.from_numpy(np.load(f'{DATA_FOLDER}/akari/{split}.npz')['data'])
        return cls(data, **_)


            