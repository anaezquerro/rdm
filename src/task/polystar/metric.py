from __future__ import annotations
import torch 
from src.trainer import Metric 
from src.util import postprocess
from .eval import PolystarEvalModel
import matplotlib.pyplot as plt 

from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image
from glob import glob
from PIL import Image
    
class PolystarMetric(Metric):
    ATTRIBUTES = ['npolygon', 'nstar', 'polygon', 'star']
    METRICS = ['POLYGON', 'STAR']
    BETTER = [1, 1]

    
    def __call__(
        self,
        *display,
        pred: torch.Tensor, 
        **kwargs
    ) -> PolystarMetric:
        """Evaluation metric for the PolyStar dataset.
        
        Args:
            pred (torch.Tensor ~ [batch_size, 3, height, width]): Generated image.
            img (torch.Tensor ~ [batch_size, 3, height, width]): Real image.
        """
        shape, n, k, digit1, digit2 = self.model.predict(pred.clamp(-1,1)).unbind(-1)
        acc =(n != 0) & (k != 0)
        acc &= ((n == digit1) | (n == digit2)) & ((k == digit1) | (k== digit2))
        self.star += (acc & (shape == 1)).sum()
        self.polygon += (acc & (shape == 0)).sum()
        self.nstar += shape.sum()
        self.npolygon += (shape == 0).sum()
        
        if display:
            kwargs |= dict(n=n, k=k, digit1=digit1, digit2=digit2)
            self.display(*display, pred=pred, acc=acc, **kwargs)
        return self
    
    def display(
        self,
        step: int, 
        path: str,
        pred: torch.Tensor,
        acc: torch.Tensor, 
        n: torch.Tensor,
        k: torch.Tensor,
        digit1: torch.Tensor, 
        digit2: torch.Tensor,
        solution: torch.Tensor,
        **_
    ):
        for i in range(pred.shape[0]):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(postprocess(pred[i]))
            ax[0].title.set_text('Reconstruction')
            ax[1].imshow(to_pil_image(solution[i]))
            ax[1].title.set_text('Real')
            fig.suptitle(f'ACC={acc[i].item()}: n={n[i].item()}, k={k[i].item()}, digit1={digit1[i].item()}, digit2={digit2[i].item()}')
            for a in ax.flatten():
                a.axis('off')
            fig.savefig(f'{path}/{step}-{i}.jpg', bbox_inches='tight')
            plt.close(fig)
        
    @property
    def STAR(self) -> float:
        return float(self.star/self.nstar)*100

    @property
    def POLYGON(self) -> float:
        return float(self.polygon/self.npolygon)*100
        
    @classmethod
    def from_model(cls, path: str) -> PolystarMetric:
        model = PolystarEvalModel()
        model.load_state_dict(torch.load(path, map_location='cuda', weights_only=True)['model'])
        return cls(model)
        
    def evaluate(self, folder: str, batch_size: int, device: str = 'cuda'):
        self.to(device)
        f = lambda path: pil_to_tensor(Image.open(path).convert('RGB').resize((128, 128)))/255*2-1
        data = torch.stack(list(map(f, sorted(glob(f'{folder}/*.png')))))
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size].to(device)
            self(pred=batch)
        return self


