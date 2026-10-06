from __future__ import annotations

import torch
import matplotlib.pyplot as plt 
from torch.nn.functional import mse_loss
from itertools import product
from torchvision.transforms.v2.functional import to_pil_image

from .akari import Akari
from src.util import postprocess
from src.trainer import Metric 

class AkariMetric(Metric):
    ATTRIBUTES = ['correct', 'total', 'unlight', 'dist', 'overlap', 'acc', 'bacc']
    METRICS = ['ACC', 'bACC', 'UNLIGHT', 'DIST', 'OVERLAP']
    KEY_METRICS = METRICS 
    BETTER = [1, 1, -1, -1, -1]

    def __init__(
        self, 
        LIGHT: torch.Tensor, 
        BULB: torch.Tensor,
        DIGIT: torch.Tensor,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.LIGHT = LIGHT
        self.BULB = BULB 
        self.DIGIT = DIGIT
        LIGHT = LIGHT.clone().view(-1, 1, 1).expand_as(BULB)
        WHITE = torch.full_like(LIGHT, 255)
        self.data = torch.stack([BULB, LIGHT, WHITE] + DIGIT, dim=0)/255*2-1

    def to(self, device: torch.device):
        super().to(device)
        self.data = self.data.to(device)
        return self

    def __call__(
        self, 
        *display,
        pred: torch.Tensor,
        digit: torch.Tensor,
        wall: torch.Tensor,
        **_ 
    ) -> AkariMetric:
        pbulb = self.detect(pred) == 0
        evals = torch.stack(list(map(Akari.eval, wall, pbulb, digit)))
        self.unlight += evals[:, 0].sum()
        self.overlap += evals[:, 1].sum()
        self.dist += evals[:, 2].sum()
        self.acc += (evals.sum(-1) == 0).sum()
        self.bacc += (evals[:, :-1].sum(-1) == 0).sum()
        self.correct += (evals == 2).sum()
        self.total += pred.shape[0]
        
        if display:
            self.display(*display, pred=pred, evals=evals, pbulb=pbulb, digit=digit, wall=wall, **_)
        return self 
    
    def detect(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect bulbs from predicted images.

        Args:
            img (torch.Tensor ~ [batch_size, 3, height, width]): Input samples.

        Returns: 
            torch.Tensor ~ [batch_size, grid_size, grid_size]: Predicted bulbs.
        """
        h, w = img.shape[-2:]
        nrow, ncol = Akari.GRID_SIZE
        cells = img.unfold(2, h//nrow, w//ncol).unfold(3, h//nrow, w//ncol)
        cells = cells.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, h//nrow, w//ncol)
        pred = mse_loss(cells.unsqueeze(1), self.data.unsqueeze(0), reduction='none').flatten(2,-1).mean(-1).argmin(-1)
        pred = pred.reshape(img.shape[0], *Akari.GRID_SIZE)
        return pred
        
    def display(
        self,
        step: int, 
        path: str,
        pred: torch.Tensor,
        pbulb: torch.Tensor,
        wall: torch.Tensor,
        digit: torch.Tensor,
        evals: torch.Tensor,
        img: torch.Tensor,
        **_
    ):
        for i in range(pred.shape[0]):
            fig, ax = plt.subplots(1, 3, figsize=(3*5, 5))
            ax[0].title.set_text('Generation')
            ax[0].imshow(postprocess(pred[i]))
            ax[1].title.set_text('Prediction')
            ax[1].imshow(self.render(wall[i], pbulb[i], digit[i]))
            ax[2].title.set_text('Real')
            ax[2].imshow(postprocess(img[i]))
            fig.suptitle(f'ACC={(evals[i].sum() == 0).item()}, bACC={(evals[i, :-1].sum() == 0).item()}')
            for a in ax.flatten():
                a.axis('off')
            fig.savefig(f'{path}/{step}-{i}.jpg', bbox_inches='tight')
            plt.close(fig)

    def render(self, wall: torch.Tensor, bulb: torch.Tensor, digit: torch.Tensor) -> torch.Tensor:
        H, W = Akari.GRID_SIZE
        C = 18
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
        return to_pil_image(img.contiguous())
    
    @property
    def ACC(self) -> float:
        return (self.acc/self.total)*100
    
    @property
    def bACC(self) -> float:
        return float(self.bacc/self.total)*100

    @property
    def DIST(self) -> float:
        return float(self.dist/self.total)
    
    @property
    def UNLIGHT(self) -> float:
        return float(self.unlight/self.total)
    
    @property
    def OVERLAP(self) -> float:
        return float(self.overlap/self.total)

class AkariReconstructionMetric(AkariMetric):
    ATTRIBUTES = ['acc', 'total']
    METRICS = ['ACC']
    BETTER = [1]


    def __call__(
        self, 
        *display,
        pred: torch.Tensor,
        target: torch.Tensor,
        **_ 
    ) -> AkariMetric:
        opred = self.detect(pred.clamp(-1, 1))
        otarget = self.detect(target)
        self.acc += (opred == otarget).sum()
        self.total += otarget.numel()
        
        if display:
            self.display(*display, pred=pred, target=target, opred=opred, otarget=otarget)
        return self 

    @property
    def ACC(self) -> float:
        return (self.acc/self.total)*100
    
    def display(
        self,
        step: int, 
        path: str,
        pred: torch.Tensor,
        target: torch.Tensor,
        opred: torch.Tensor, 
        otarget: torch.Tensor,
        **_
    ):
        for i in range(pred.shape[0]):
            fig, ax = plt.subplots(1,2 , figsize=(2*5, 5))
            ax[0].title.set_text('Reconstruction')
            ax[0].imshow(postprocess(pred[i]))
            ax[1].title.set_text('Real')
            ax[1].imshow(postprocess(target[i]))
            fig.suptitle(f'ACC={(opred[i] == otarget[i]).float().mean().item()*100:.2f}')
            for a in ax.flatten():
                a.axis('off')
            fig.savefig(f'{path}/{step}-{i}.jpg', bbox_inches='tight')
            plt.close(fig)
