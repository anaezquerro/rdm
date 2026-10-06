
from __future__ import annotations
from src.trainer.metric import Metric
import torch
import matplotlib.pyplot as plt 
from src.util import postprocess, batch_quantize, hex_to_tensor
from torchvision.transforms.v2.functional import to_pil_image
from .tangram import Tangram 


class TangramMetric(Metric):
    ATTRIBUTES = ['correct', 'shape', 'parallelogram', 'square', 'striangle', 'mtriangle', 'ltriangle', 'total']
    METRICS = ['ACC', 'sACC', 'PARALLELOGRAM', 'SQUARE', 'STRIANGLE', 'MTRIANGLE', 'LTRIANGLE']
    BETTER = [1 for _ in METRICS]
    COLORS = hex_to_tensor(['#000000', *Tangram.COLORS])

    def __call__(
        self, 
        *display,
        pred: torch.Tensor, 
        layout: torch.Tensor,
        tangram: torch.Tensor, 
        color: torch.Tensor,
        **_
    ) -> TangramMetric:
        """Evaluate a batch of Tangram images.
        
        Args:
            pred (torch.Tensor ~ [batch_size, *img_size]): Generated Tangram solution.
            layout (torch.Tensor ~ [batch_size, 1, *resolution]): Tangram shilouette.
            tangram (torch.Tensor ~ [batch_size, 7, *resolution]): Tangram binary masks.
            color (torch.Tensor ~ [batch_size, 7, 3, 1, 1]): Piece colors.

        Returns:
            TangramMetric: Updated Tangram metric.
        """
        # quantize the prediction to only allow the specified colors
        qpred = batch_quantize(pred, self.COLORS.to(pred.device))[0]
        # apply the silhouette as a mask
        mask = tangram.any(1)
        boundary = ((qpred.mean(1) == -1) == ~mask).float().flatten(1,-1).mean(-1)

       # get the piece-wise result
        default = (0, torch.zeros(5, device=mask.device), 0)
        result, score = zip(*[Tangram.evaluate(qpred[b],  color[b]) if boundary[b] > 0.7 else default for b in range(qpred.shape[0])])
        result = torch.tensor(result, device=pred.device)
        score = torch.stack(score)
        self.correct += (result == 2).sum()
        self.shape += (result > 0).sum()
        self.parallelogram += score[:, 0].sum()
        self.square += score[:, 1].sum()
        self.mtriangle += score[:, 2].sum()
        self.striangle += score[:, 3].sum()
        self.ltriangle += score[:, 4].sum()
        self.total += pred.shape[0]
        if display:
            self.display(*display, pred=pred, result=result, layout=layout, tangram=tangram, color=color, **_)
        return self
    
    def display(
        self,
        step: int, 
        path: str, 
        pred: torch.Tensor,
        solution: torch.Tensor, 
        layout: torch.Tensor,
        piece: torch.Tensor,
        result: torch.Tensor,
        tangram: torch.Tensor,
        color: torch.Tensor,
        **_
    ):
        for i in range(pred.shape[0]):
            if result[i] > 0:
                continue 
            fig, ax = plt.subplots(1, 4, figsize=(4*5, 5))
            ax[0].imshow(to_pil_image(solution[i]))
            ax[0].title.set_text('Solution')
            ax[1].imshow(to_pil_image(piece[i]))
            ax[1].title.set_text('Pieces')
            ax[2].imshow(layout[i][0].cpu().numpy())
            ax[2].title.set_text('Layout')
            ax[3].imshow(postprocess(pred[i]))
            ax[3].title.set_text("Prediction")
            fig.suptitle(f'ACC={(result[i] == 2).all()}, rACC={(result[i] > 0).all()}')
            for a in ax.flatten():
                a.axis('off')
            fig.savefig(f'{path}/{step}-{i}.jpg', bbox_inches='tight')
            torch.save(dict(pred=pred[i], tangram=tangram[i], color=color[i]), f'{path}/{step}-{i}.pt')
            plt.close(fig)
            
    @property 
    def ACC(self) -> float:
        return float(self.correct/self.total*100)
    
    @property
    def sACC(self) -> float:
        return float(self.shape/self.total*100)

    @property
    def PARALLELOGRAM(self) -> float:
        return float(self.parallelogram/self.total*100)

    @property
    def SQUARE(self) -> float:
        return float(self.square/self.total*100)

    @property
    def MTRIANGLE(self) -> float:
        return float(self.mtriangle/self.total*100)
    
    @property
    def STRIANGLE(self) -> float:
        return float(self.striangle/self.total*100)

    @property
    def LTRIANGLE(self) -> float:
        return float(self.ltriangle/self.total*100)




class TangramReconstructionMetric(TangramMetric):
    ATTRIBUTES = ['correct', 'total']
    METRICS = ['ACC']
    BETTER = [1]

    def __call__(
        self, 
        *display,
        pred: torch.Tensor, 
        target: torch.Tensor,
        tangram: list[Tangram], 
        **_
    ) -> TangramMetric:
        """Evaluate a batch of Tangram images."""
        # quantize the prediction to only allow the specified colors
        qpred = batch_quantize(pred, self.COLORS.to(pred.device))[0]

        # apply the silhouette as a mask
        mask = torch.stack([t.mask(pred.shape[-2:]).unsqueeze(0) for t in tangram]).to(pred.device)
        boundary = ((qpred.mean(1) == -1) == ~mask).float().flatten(1,-1).mean(-1)

        # get the piece-wise result
        default = (0, torch.zeros(5, device=mask.device), 0)
        result, *__ = zip(*[t.evaluate(x) if b > 0.8 else default for x, t, b in zip(qpred.unbind(0), tangram, boundary)])
        result = torch.tensor(result, device=target.device)
        self.correct += (result == 2).sum()
        self.total += pred.shape[0]
        if display:
            self.display(*display, pred=pred, result=result, target=target, **_)
        return self
    
    def display(
        self,
        step: int, 
        path: str, 
        pred: torch.Tensor,
        target: torch.Tensor, 
        result: torch.Tensor,
        **_
    ):
        for i in range(pred.shape[0]):
            fig, ax = plt.subplots(1, 2, figsize=(2*5, 5))
            ax[0].imshow(postprocess(target[i]))
            ax[0].title.set_text('Real')
            ax[1].imshow(postprocess(pred[i]))
            ax[1].title.set_text('Reconstruction')
            fig.suptitle(f'ACC={(result[i] == 2).all()}, rACC={(result[i] > 0).all()}')
            for a in ax.flatten():
                a.axis('off')
            fig.savefig(f'{path}/{step}-{i}.jpg', bbox_inches='tight')
            plt.close(fig)
            