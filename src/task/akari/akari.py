from __future__ import annotations
import torch, random
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.v2.functional import pil_to_tensor
from torch.nn.functional import pad


from src.util import shuffle
from src.env import FONT



def draw_digit(digit: int, size: int) -> torch.Tensor:
    font = ImageFont.truetype(FONT, size=int(size*0.7))
    img = Image.new('RGB', size=(size, size), color='black')
    draw = ImageDraw.Draw(img)
    draw.text((size//3, size*0.1), str(digit), font=font, fill='white', align='center')
    return pil_to_tensor(img)



class Akari:
    GRID_SIZE = [11, 11]

    def __init__(self, wall: torch.Tensor, bulb: torch.Tensor, digit: torch.Tensor):
        """Initialization of the Akari object.
        
        Args:
            wall (torch.Tensor[bool] ~ GRID_SIZE): Wall positions.
            bulb (torch.Tensor[bool] ~ GRID_SIZE]): Bulb positions.
            clue (torch.Tensor[bool] ~ [*GRID_SIZE, 5]): Numbered cells [0-4].
        """
        self.wall = wall.bool()
        self.bulb = bulb.bool()
        self.digit = digit 

    def tensorize(self) -> torch.Tensor:
        """Returns Akari representation as binary masks.
        
        Returns:
            torch.Tensor ~ [*GRID_SIZE, 7]: Wall, bulb and digit positions.
        """
        return torch.cat([self.wall.unsqueeze(-1), self.bulb.unsqueeze(-1), self.digit], dim=-1).bool()

    def __getitem__(self, pos: tuple[int, int]) -> int:
        row, col = pos
        if row in range(self.GRID_SIZE) and col in range(self.GRID_SIZE):
            if self.bulb[row, col]:
                return 1 
            elif self.wall[row, col]:
                return -1 
            else:
                return 0
        else:
            return -1


        
    @classmethod
    def eval(cls, wall: torch.Tensor, bulb: torch.Tensor, digit: torch.Tensor) -> tuple[int, int, int]:
        """Checks whether the bulb locations are correct for the given grid.
        
        Args:
            wall (torch.Tensor ~ [GRID_SIZE, GRID_SIZE]): Boolean, tensor, indicating where walls are located.
            bulb (torch.Tensor ~ [GRID_SIZE, GRID_SIZE]): Boolean tensor, indicating where a bulb is located.
            digit (torch.Tensor ~ [GRID_SIZE, GRID_SIZE]): Integer tensor with digit clues.
            
        Returns:    
            Number of cells that have remained unlighted.
            Number of bulbs that look at each other.
            Number of cells that do not respect the wall limit.
        """
        # check that there are no bulbs in the walls
        nrow, ncol = Akari.GRID_SIZE
        bulb[wall.to(bulb.device)] = False 
        lighted = wall.to(bulb.device).clone() # assume that walls are "lighted"
        result = torch.zeros(3, device=bulb.device)
        
        # iterate over bulbs
        for row, col in bulb.nonzero().tolist():
            lighted[row, col] = True 
            # look at the right
            c = col+1
            while row in range(nrow) and c in range(ncol) and not wall[row, c]:
                if bulb[row, c]:
                    result[1] += 1 
                lighted[row, c] = True
                c += 1
            # look at the left
            c = col - 1
            while row in range(nrow) and c in range(ncol) and not wall[row, c]:
                lighted[row, c] = True 
                c -= 1
            # look below
            r = row+1
            while r in range(nrow) and col in range(ncol) and not wall[r, col]:
                if bulb[r, col]:
                    result[1] += 1
                lighted[r, col] = True 
                r += 1
            # look up 
            r = row - 1
            while r in range(nrow) and col in range(ncol) and not wall[r, col]:
                lighted[r, col] = True 
                r -= 1
        
        # check clues 
        bulb = pad(bulb, (0, 1, 0, 1), mode='constant', value=False).int()
        for row, col, value in digit.nonzero():
            num = bulb[row+1, col] + bulb[row-1, col] + bulb[row, col+1] + bulb[row, col-1]
            result[-1] += abs(value - num)
        result[0] = (~lighted).sum()
        return result 
    
    @classmethod
    def generate(cls, wall_ratio: tuple[float, float] = (0.2, 0.3), digit_ratio: float = 0.5) -> Akari:
        wall = cls.generate_wall(wall_ratio)
        bulb = cls.place_bulb(wall)
        digit = cls.get_digit(wall, bulb, digit_ratio)
        return cls(wall, bulb, digit)
        
    @classmethod
    def place_bulb(cls, wall: torch.Tensor) -> torch.Tensor:
        cells = shuffle((r, c) for r in range(cls.GRID_SIZE[0]) for c in range(cls.GRID_SIZE[1]) if not wall[r, c])
        bulb = torch.zeros_like(wall, dtype=bool)
        for r, c in cells: # loop empty cells
            # check if this bulb can see another bulb
            can_see_bulb = False
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                while 0 <= nr < cls.GRID_SIZE[0] and 0 <= nc < cls.GRID_SIZE[1]:
                    if bulb[nr, nc]:
                        can_see_bulb = True
                        break
                    if wall[nr, nc]:
                        break
                    nr += dr
                    nc += dc
            if not can_see_bulb:
                bulb[r, c] = True
        return bulb.bool()
                    
    @classmethod
    def generate_wall(cls, wall_ratio: tuple[float, float]) -> torch.Tensor:
        wall = torch.zeros(*cls.GRID_SIZE, dtype=bool).flatten()
        num = int(random.uniform(*wall_ratio)*wall.numel())
        indices = torch.randperm(wall.numel())[:num]
        wall[indices] = True 
        return wall.reshape(cls.GRID_SIZE)
    
    @classmethod
    def get_digit(cls, wall: torch.Tensor, bulb: torch.Tensor, ratio: float) -> torch.Tensor:            
        clues = torch.zeros(*wall.shape, 5).bool()
        for r in range(cls.GRID_SIZE[0]):
            for c in range(cls.GRID_SIZE[1]):
                if wall[r, c]:
                    count = 0
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        if (0 <= r+dr < cls.GRID_SIZE[0]) and (0 <= c+dc < cls.GRID_SIZE[1]) and bulb[r+dr, c+dc]:
                            count += 1
                    if random.uniform(0, 1) < ratio:
                        clues[r, c, count] = True
        return clues
    