
from __future__ import annotations
from typing  import Callable
from torch.nn.functional import interpolate
from torch import nn 
from PIL import Image, ImageDraw, ImageFont
import torch 

import matplotlib.pyplot as plt 
from .eval import MNISTModel, MNISTDataset
from src.util import postprocess, nunique
from src.trainer import Metric, AccuracyMetric
from torch.nn.functional import interpolate

class SudokuMetric(Metric):
    ATTRIBUTES = ['digit', 'color', 'correct', 'total']
    METRICS = ['DIGIT', 'COLOR', 'ACC']
    BETTER = [1, 1, 1]
    COLORS = torch.tensor([
        *(MNISTDataset.COLORS/255*2-1).tolist(),
        [-1.0, -1.0, -1.0],
    ])

    def __init__(self, model: nn.Module, postprocess: Callable):
        super().__init__(model)
        self.postprocess = postprocess 

    def __call__(
        self,
        *display, 
        pred: torch.Tensor,
        digit: torch.Tensor,
        color: torch.Tensor,
        mask: torch.Tensor, 
        img: torch.Tensor,
        meta: dict = None,
        **_
    ) -> SudokuMetric:
        """Evaluates a batch of Sudoku images.

        Args:
            imgs (torch.Tensor[float] ~ [batch_size, 3, height, width]): Input Sudoku images.

        Returns:
            list[dict[str, float]]: Sample results.
        """
        if self.postprocess is not None:
            img = torch.stack(list(map(self.postprocess, img, meta)))
            bpred = torch.stack(list(map(self.postprocess, pred, meta))) # bilinear interpolation
            epred = torch.stack([self.postprocess(p, m, exact=True) for p, m in zip(pred, meta)]) # exact interpolation
            dpred, _ = self.detect(bpred.clamp(-1, 1), mode='bilinear')
            _, cpred = self.detect(epred.clamp(-1, 1), mode='nearest')
        else:
            dpred, _ = self.detect(pred.clamp(-1,1), mode='bilinear')
            _, cpred = self.detect(pred.clamp(-1,1), mode='nearest')

        dpred = torch.where(mask, digit, dpred)
        cpred = torch.where(mask, color, cpred)
        dmask = self.eval_grids(dpred.clone()).sum(-1) == 0
        cmask = self.eval_grids(cpred.clone()).sum(-1) == 0
        self.digit += dmask.sum().detach()
        self.color += cmask.sum().detach()
        self.correct += (dmask & cmask).sum().detach()
        self.total += pred.shape[0]
        
        if display:
            self.display(*display, pred=pred, dacc=dmask, cacc=cmask, dpred=dpred, cpred=cpred, img=img, mask=mask)
        return self 
    
    def display(
        self, 
        step: int,
        path: str, 
        pred: torch.Tensor,
        img: torch.Tensor,
        mask: torch.Tensor, 
        dpred: torch.Tensor,
        cpred: torch.Tensor,
        dacc: torch.Tensor, 
        cacc: torch.Tensor,
        **_
    ):
        spatial_mask = interpolate(mask.float().unsqueeze(1), size=img.shape[-2:], mode='nearest')
        clue = torch.where(spatial_mask.bool(), img, -1)
        for i in range(pred.shape[0]):
            fig, ax = plt.subplots(1, 4, figsize=(4*5, 5))
            ax[0].imshow(postprocess(clue[i]))
            ax[0].title.set_text('Input')
            ax[1].imshow(postprocess(pred[i]))
            ax[1].title.set_text('Predicted')
            ax[2].imshow(self.render(dpred[i], cpred[i]))
            ax[2].title.set_text('Renderized')
            ax[3].imshow(postprocess(img[i]))
            ax[3].title.set_text('Target')
            fig.suptitle(f'dACC={dacc[i].item()}, cACC={cacc[i].item()}')
            fig.savefig(f'{path}/{step}-{i}.jpg', bbox_inches='tight')
            plt.close()
    
        
    def detect(self, img: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect the Sudoku digits.
        
        Args:
            imgs (torch.Tensor[float] ~ [batch_size, 3, height, width]): Input Sudoku images.

        Returns:
            torch.Tensor ~ [batch_size, 9, 9]: Sudoku digits.
            torch.Tensor ~ [batch_size, 9, 9]: Digit colors.
        """
        batch_size, num_channels = img.shape[:2]
        if num_channels == 1:
            img = img.repeat(1, 3, 1, 1)
        img = interpolate(img, size=(28*9, 28*9), mode=mode, align_corners=False if mode == 'bilinear' else None)
        flat = img.reshape(-1, 3, 9, 28, 9, 28).permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, 28, 28)
        s_digit, s_color = self.model(flat.to(torch.float32))
        dpred = s_digit.argmax(-1).reshape(batch_size, 9, 9)
        cpred = s_color.argmax(-1).reshape(batch_size, 9, 9)
        return dpred, cpred
        
    def eval_grids(self, sudokus: torch.Tensor) -> torch.Tensor:
        """Evaluation of a batch of Sudokus.
        
        Args:  
            sudokus (torch.Tensor[float] ~ [batch_size, 9, 9]): Input Sudokus.
            
        Returns:
            torch.Tensor ~ [batch_size, 3]: Row, column and block error.
        """
        batch_size = sudokus.shape[0]

        # measure row, column and block errors
        row = (9-nunique(sudokus, dim=-1)).sum(-1)
        col = (9-nunique(sudokus, dim=-2)).sum(-1)
        block = sudokus.view(batch_size, 3, 3, 3, 3).permute(0,1,3,2,4).contiguous().view(sudokus.shape)
        block = (9-nunique(block, dim=-1)).sum(-1)

        return torch.stack([row, col, block], dim=-1)

    def copy(self) -> SudokuMetric:
        new = self.__class__(self.model)
        for attr in self.ATTRIBUTES:
            setattr(new, attr, self.__getattribute__(attr))
        return new

    @property
    def DIGIT(self) -> float:
        return float(self.digit/self.total*100)
    
    @property
    def COLOR(self) -> float:
        return float(self.color/self.total*100)
    
    @property
    def ACC(self) -> float:
        return float(self.correct/self.total*100)
    
    @classmethod 
    def from_model(cls, path: str, post: Callable) -> SudokuMetric:
        model = MNISTModel()
        model.load_state_dict(torch.load(path, map_location='cuda', weights_only=True)['model'])
        return cls(model, post)

    def render(
        self,
        digit: torch.Tensor,
        color: torch.Tensor,
        resolution: tuple[int, int] = (500, 500)
    ) -> Image:
        """Renders the image of a Colored Sudoku from the tensor predictions.
        
        Args:
            digit (torch.Tensor ~ [9, 9]): Digit values (from 1 to 9).
            color (torch.Tensor ~ [9, 9]): Color values (from 1 to 9).

        Returns:
            PIL.Image: Renderized image.
        """
        h, w = resolution
        img = Image.new("RGB", (w, h), (0, 0, 0)) # black background
        draw = ImageDraw.Draw(img)
        cell_h, cell_w = h / 9, w / 9

        font = ImageFont.load_default(size=int(h/9*0.8))

        for r in range(9):
            for c in range(9):
                text = str(digit[r, c].item())
                clr = MNISTDataset.COLORS[color[r,c]].tolist()
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                pos_x = c * cell_w + (cell_w - text_w) / 2
                pos_y = r * cell_h + (cell_h - text_h) / 2 - (bbox[1]) 
                draw.text((pos_x, pos_y), text, fill=tuple(clr), font=font)
        return img

    def save(self, path: str):
        delattr(self, 'postprocess')
        super().save(path)

class SudokuReconstructionMetric(AccuracyMetric, SudokuMetric):
            
    def __call__(
        self, 
        *display, 
        pred: torch.Tensor, 
        digit: torch.Tensor, 
        color: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        meta: dict = None,
        **_
    ) -> SudokuMetric:
        if self.postprocess is not None:
            pred = torch.stack(list(map(self.postprocess, pred, meta))) 
        dpred, cpred = self.detect(pred.clamp(-1,1), mode='bilinear')
        acc = (dpred[mask] == digit[mask]) & (cpred[mask] == color[mask])
        self.correct += acc.sum().detach()
        self.total += mask.sum()
        if display:
            self.display(*display, pred=pred, acc=acc.split(mask.flatten(1,-1).sum(-1).tolist()), target=target, **_)
        return self
    
    def display(
        self,
        step: int,
        path: str,
        pred: torch.Tensor, 
        target: torch.Tensor, 
        acc: torch.tensor,
        **_
    ):
        for i in range(pred.shape[0]):
            fig, ax = plt.subplots(1, 2, figsize=(2*5, 5))
            ax[0].imshow(postprocess(pred[i]))
            ax[0].title.set_text('Reconstructed')
            ax[1].imshow(postprocess(target[i]))
            ax[1].title.set_text('Real')
            fig.suptitle(f'ACC={acc[i].float().mean()*100:.2f}')
            for a in ax.flatten():
                a.axis('off')
            fig.savefig(f'{path}/{step}-{i}.jpg', bbox_inches='tight')
            plt.close(fig)
