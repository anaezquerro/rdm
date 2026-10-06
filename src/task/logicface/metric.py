from __future__ import annotations
from src.trainer import Metric 
from .eval import CelebAModel, CelebAMetric
import torch
from torch.nn.functional import avg_pool1d, interpolate
import matplotlib.pyplot as plt
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from src.util import postprocess
    
class LogicFaceMetric(Metric):
    METRICS = ['ACC', 'fACC', 'FID']
    BETTER = [1, 1]
    ATTRIBUTES = ['correct', 'total', 'tp']
    
    def __init__(self, model: CelebAModel):
        super(LogicFaceMetric, self).__init__(model)
        self.tp = torch.zeros(4)
        self.fid = FrechetInceptionDistance(feature=2048)

    def to(self, device: torch.device):
        self.fid.to(device)
        return super().to(device)
                
    def __call__(
        self,
        *display,
        pred: torch.Tensor, # [batch_size, 3, height, width]
        attr: torch.Tensor, # [batch_size, 3, num_feats]
        img: torch.Tensor,
        **_
    ) -> LogicFaceMetric:
        self.total += pred.shape[0]
        # predict the features of the generated image 
        fpred = self.model(img.reshape(-1, 3, 128, 128).clamp(-1,1)).reshape(-1, 3, 4) > 0
        out = self.model(pred.clamp(-1, 1)) > 0
        
        # asses the correctness with the operators 
        acc = torch.ones(attr.shape[0], attr.shape[-1], device=attr.device, dtype=bool)
        for i, op in enumerate(self.OPERATORS):
            acc[:, i] &= op.eval(attr[:, 0, i], attr[:, 1, i], out[..., i])
        self.correct += acc.all(-1).sum()
        self.tp += acc.sum(0)
        
        # update fid score
        self.fid.update(postprocess(img[:, -3:], to_pil=False), real=True)
        self.fid.update(postprocess(pred, to_pil=False), real=False)

        if display:
            self.display(*display, pred=pred, img=img, out=out, attr=attr, fpred=fpred, acc=acc.all(-1), **_)
        return self
    
    
    def display(
        self,
        step: int,
        path: str,
        img: torch.Tensor, # [batch_size, 3, height, width]
        pred: torch.Tensor, # [batch_size, 3, height, width]
        out: torch.Tensor, # [batch_size, num_feats]
        attr: torch.Tensor, # [batch_size, 3, num_feats]
        fpred: torch.Tensor, # [batch_size, 3, num_feats]
        depth_mask: torch.Tensor, # [batch_size, 3],
        acc: torch.Tensor,
        **_
    ):
        """Display the result and store it in a folder."""
        pmask = avg_pool1d((~depth_mask).float(), kernel_size=3).bool()
        for i in range(img.shape[0]):
            fig, ax = plt.subplots(2, 3, figsize=(3*5, 2*5))
            A, B, S = img[i].chunk(3, dim=0)
            ax[0,0].imshow(postprocess(A))
            ax[0,1].imshow(postprocess(B))
            ax[0,2].imshow(postprocess(S))
            ax[1,0].imshow(postprocess(A))
            ax[1,1].imshow(postprocess(B))
            ax[1,2].imshow(postprocess(S))
            for j in range(3):
                ax[0, j].title.set_text(f'Annotations: {attr[i,j].int().tolist()}\nClassifier: {fpred[i,j].int().tolist()}')

            j = pmask[i].nonzero().item()
            ax[1,j].imshow(postprocess(pred[i]))
            ax[1,j].title.set_text(out[i].int().tolist())
            
            for a in ax.flatten():
                a.axis('off')
            fig.suptitle(f'[acc={acc[i].int()}] Features: ' + ', '.join(
                f'{feat} ({repr(op)})' for feat, op in zip(self.FEATS, self.OPERATORS)))
            fig.savefig(f'{path}/{step}-{i}.jpg', bbox_inches='tight')
            plt.close()
        
    @property
    def ACC(self) -> float:
        return float(self.correct/self.total*100)

    @property
    def fACC(self) -> float:
        return float((self.tp/self.total).mean()*100)

    @property
    def smiling(self) -> float:
        return float(self.tp[0]/self.total*100)
    
    @property
    def male(self) -> float:
        return float(self.tp[1]/self.total*100)

    @property
    def eyeglasses(self) -> float:
        return float(self.tp[2]/self.total*100)

    @property
    def young(self) -> float:
        return float(self.tp[3]/self.total*100)

    @property
    def FID(self) -> float:
        return float(self.fid.compute())
    
    @classmethod 
    def from_model(cls, path: str) -> LogicFaceMetric:
        model = CelebAModel()
        model.load_state_dict(torch.load(path, map_location='cuda', weights_only=False)['model'])
        return cls(model)
        
        
class LogicFaceReconstructionMetric(CelebAMetric):
    
    def __call__(
        self, 
        *display,
        pred: torch.Tensor,
        target: torch.Tensor,
        attr: torch.Tensor,
        img: torch.Tensor,
        **_
    ) -> list[dict[str, float]]:
        fpred = self.model(pred.clamp(-1, 1)) > 0
        ftarget = self.model(target.clamp(-1, 1)) > 0
        super().__call__(pred=fpred, target=ftarget, img=img)
        if display:
            self.display(*display, pred=pred, target=target, fpred=fpred, ftarget=ftarget, attr=attr)

    def display(
        self, 
        step: int, 
        path: str, 
        pred: torch.Tensor,
        target: torch.Tensor,
        fpred: torch.Tensor, 
        ftarget: torch.Tensor,
        attr: torch.Tensor,
        **_
    ):
        for i in range(pred.shape[0]):
            fig, ax = plt.subplots(1, 2, figsize=(10,5))
            ax[0].imshow(postprocess(pred[i]))
            ax[0].title.set_text(f'Reconstructed:\n{fpred[i].int().tolist()}')
            ax[1].imshow(postprocess(target[i]))
            ax[1].title.set_text(f'Real: {attr[i].int().tolist()}\n{ftarget[i].int().tolist()}')
            fig.suptitle(f'Features: ' + ', '.join(self.FEATS))
            for a in ax.flatten():
                a.axis('off')
            fig.savefig(f'{path}/{step}-{i}.jpg', bbox_inches='tight')
            plt.close(fig)

    @classmethod 
    def from_model(cls, path: str) -> LogicFaceMetric:
        model = CelebAModel()
        model.load_state_dict(torch.load(path, map_location='cuda', weights_only=False)['model'])
        return cls(model)
        