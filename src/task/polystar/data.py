from __future__ import annotations

import random, torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.v2.functional import pil_to_tensor

from src.env import DATA_FOLDER, EVAL_FOLDER, FONT
from src.trainer import Dataset, AccuracyMetric
from .metric import PolystarMetric

class PolystarDataset(Dataset):
    RESOLUTION: tuple[int, int] = (128, 128)
    NUM_CHANNELS = dict(solution=3)

    def __init__(self, color: torch.Tensor, points: dict[str, np.ndarray], shape: str, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape 
        self.points = points
        self.color = color 
        self.draw = dict(polygon=self.draw_polygon, star=self.draw_star)
        self.side = dict(polygon=(3, 9), star=(2, 7))

    def sample_shape(self):
        if isinstance(self.shape, str):
            shape = self.shape 
        else:
            shape = random.choice(['polygon', 'star'])
        return shape 
        
    def OL(self, i: int) -> dict[str, torch.Tensor]:
        return self.RS(i)
    
    def RS(self, i: int) -> dict[str, torch.Tensor]:
        face = self.data[i]
        color = self.color[i]
        # sample values for (n, k)
        n = random.randint(1, 9)
        shape = self.sample_shape()
        k = random.randint(*self.side[shape])
        background = Image.new('L', self.RESOLUTION, 0)
        draw = ImageDraw.Draw(background)
        point = sorted((self.points[n+2][i]*self.RESOLUTION[0]).astype(int), key=lambda x: x[-1])
        for x, y, r in point[:-2]:
            self.draw[shape](draw, x, y, r, k)
        self.draw_digit(draw, *point[-2], k)
        self.draw_digit(draw, *point[-1], n)
        mask = (pil_to_tensor(background) > 127).squeeze(0)
        solution = torch.where(~mask, face, color.view(-1, 1, 1)).to(torch.uint8)
        digit = torch.zeros(10, dtype=int)
        digit[n] = 1
        digit[k] = 1
        return dict(solution=solution, meta=torch.tensor([int(shape == 'star'), n, k, *digit.tolist()]))

    def CF(self, i: int) -> dict[str, torch.Tensor]:
        face = self.data[i]
        color = self.color[i]
        n = random.randint(1, 9)
        shape = self.sample_shape()
        k = random.randint(*self.side[shape])
        background = Image.new('L', self.RESOLUTION, 0)
        draw = ImageDraw.Draw(background)
        point = sorted((self.points[n+2][i]*self.RESOLUTION[0]).astype(int), key=lambda x: x[-1])
        option = random.randint(0, 2)
        if option == 0: # create shapes with different number of sides 
            k = 0
            for x, y, r in point[:-2]:
                self.draw[shape](draw, x, y, r, random.randint(*self.side[shape]))
        elif option == 1: # do not display any shape 
            n, k = 0, 0
        else: # normal behavior
            for x, y, r in point[:-2]:
                self.draw[shape](draw, x, y, r, k)

        digit = torch.zeros(10, dtype=int)
        if random.uniform(0, 1) < 0.5:
            num = random.randint(1, 9)
            self.draw_digit(draw, *point[-2], num)
            digit[num] = 1 
        if random.uniform(0, 1) < 0.5:
            num = random.randint(1, 9)
            self.draw_digit(draw, *point[-1], num)
            digit[num] = 1

        mask = (pil_to_tensor(background) > 127).squeeze(0)
        solution = torch.where(~mask, face, color.view(-1, 1, 1)).to(torch.uint8)
        return dict(solution=solution, meta=torch.tensor([int(shape=='star'), n, k, *digit.tolist()]))

            
    def draw_polygon(self, draw: ImageDraw, x: int, y: int, r: int, k: int):
        # generate vertices
        angle = 2*np.pi*np.arange(k)/k
        vx = x + r*np.cos(angle)
        vy = y + r*np.sin(angle)

        # obtain random rotation
        theta = random.uniform(0, 2 * np.pi)
        xr = x + (vx - x) * np.cos(theta) - (vy - y) * np.sin(theta)
        yr = y + (vx - x) * np.sin(theta) + (vy - y) * np.cos(theta)

        draw.polygon(list(zip(xr, yr)), fill='white')

    def draw_star(self, draw: ImageDraw, x: int, y: int, r: int, k: int):
        inner = int(r * 0.2)
        radius = np.array([r if v % 2 == 0 else inner for v in range(2 * k)])
        theta = random.uniform(0, 2 * np.pi)
        angles = np.linspace(0, 2 * np.pi, 2 * k, endpoint=False) + theta
        xr = x + radius * np.cos(angles)
        yr = y + radius * np.sin(angles)
        points = list(zip(xr.tolist(), yr.tolist()))
        draw.polygon(points, fill="white")
            
    def draw_digit(self, draw: ImageDraw, x: int, y: int, r: int, digit: int):
        font = ImageFont.truetype(FONT, r*1.5)
        text = str(digit)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        txt_layer = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_layer)
        txt_draw.text((-bbox[0], -bbox[1]), text, font=font, fill="white")

        rotated_txt = txt_layer.rotate(0, expand=True, resample=Image.BICUBIC)
        draw._image.paste(rotated_txt, (x, y), rotated_txt)


    @property
    def METRIC(self) -> PolystarMetric:
        if self.mode == 'CF':
            return AccuracyMetric()
        else:
            return PolystarMetric.from_model(f'{EVAL_FOLDER}/detector/last.pt')

    @classmethod
    def from_folder(cls, **_) -> PolystarDataset:
        data = np.load(f'{DATA_FOLDER}/polystar/data.npz', allow_pickle=False)
        points = np.load(f'{DATA_FOLDER}/circle_position_radius.npy', allow_pickle=True).item()
        images = torch.from_numpy(data['images']).movedim(-1,1)
        color = torch.from_numpy(data['color'])
        return cls(data=images, color=color, points=points, **_)