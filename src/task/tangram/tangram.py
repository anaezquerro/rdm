from __future__ import annotations
import torch, cv2
import numpy as np 
from torchvision.transforms.v2.functional import pil_to_tensor

from .polygon import Polygon
from .svg import SVG 
from src.util import shuffle


class Tangram(SVG):
    TYPES = ['parallelogram', 'square', 'triangle']
    PIECES = ['parallelogram', 'square', 'striangle1', 'striangle2', 'mtriangle', 'ltriangle1', 'ltriangle2']
    COLORS = (
        '#FF0000',
        '#00FF00',
        '#0000FF',
        '#FFFF00',
        '#FF00FF',
        '#00FFFF',
        '#663399',
        '#FF8C00',
        '#008080',
    )

    
    def __init__(self, polygons: list[Polygon], viewbox: tuple[float, float, float, float] = None):
        super(Tangram, self).__init__(polygons, viewbox)
        self.pieces = dict(zip(self.PIECES, self.polygons))

    @property
    def whiten(self) -> Tangram:
        new = self.copy()
        for polygon in new.polygons:
            polygon.color = '#FFFFFF'
        return new

    def mask(self, size: tuple[int, int]) -> torch.Tensor:
        """Compute a tensor mask of the Tangram layout. Required
        since some polygon borders are not fully aligned, leaving
        some black pixels between each other.
        
        Args: 
            size (tuple[int, int]): Mask resolution.
        
        Returns: 
            torch.Tensor ~ [height, width]: Boolean mask of the layout.
        """
        mask = np.array(self.whiten.to_img().resize(size).convert('L'))
        # apply closing operation to close black pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return torch.from_numpy(mask).squeeze(0) > 127


    @classmethod
    def evaluate(cls, img: torch.Tensor, color: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Checks whether the input image corresponds to the same tangram.
        
        Args:
            img (torch.Tensor ~ [3, height, width]): Input image (assumed quantized).
            
        Returns:
            (int) Whether the Tangram is correct (2), fulfills the tan pieces (1), or is incorrect (0).
            (torch.Tensor ~ [5]): F-score of Tangram piece type.
        """
        img = (((img+1)/2)*255).int().clamp(0, 255)
        pred = SVG.from_generated(img.permute(1,2,0).detach().cpu().numpy(), hex_colors=cls.COLORS)
        if pred is None:
            return 0, torch.zeros(5, dtype=int, device=img.device), 0

        # remove overlapping polygons, preference to the real colors
        color = [cls.COLORS[c] for c in color]
        pred = pred.remove_overlap(ratio=0.5, colors=color)

        if pred is None or not pred.is_tangram():
            return 0, torch.zeros(5, dtype=int, device=img.device)
        
        # check each piece
        pred = Tangram(pred.polygons)
        score = torch.zeros(5, device=img.device)

        score[0] += pred.pieces['parallelogram'].color == color[0]
        score[1] += pred.pieces['square'].color == color[1]
        score[2] += pred.pieces['mtriangle'].color == color[4]

        real_colors = set(color[2:4])
        pred_colors = {pred.pieces[f'striangle1'].color, pred.pieces[f'striangle2'].color}
        score[3] += real_colors == pred_colors

        real_colors = set(color[5:])
        pred_colors = {pred.pieces[f'ltriangle1'].color, pred.pieces[f'ltriangle2'].color}
        score[4] += real_colors == pred_colors
        return 2 if (score > 0).all() else 1, score.int()
    
    def get_sample(self, resolution: tuple[int, int]) -> dict[str, torch.Tensor]:
        colored = self.random_affine(scale=0.8).coloring(shuffle(self.COLORS)[:7])
        piece = self.shuffle()
        layout = colored.mask(resolution)
        return dict(
            img=pil_to_tensor(colored.to_img().resize(resolution)),
            piece=pil_to_tensor(piece.to_img().resize(resolution)),
            layout=layout.bool(),
        ) | colored.content

    def tensorize(self, size: tuple[int, int]) -> torch.Tensor:
        """Returns the tensorized version of the Tangram.
        
        Args:
            size (tuple[int, int]): Output resolution.

        Returns:
            torch.Tensor ~ [7, *size]: Tangram solution (one channel per piece).
        """
        x = [pil_to_tensor(SVG([p], self.viewbox).to_img().resize(size).convert('L')) for p in self.polygons]
        return torch.stack(x).bool().view(-1, *size)

        