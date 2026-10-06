from transformers import AutoModel
import torch 
from torch import nn 

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        pretrained: str, 
        **_
    ): 
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained).requires_grad_(False)
        self.init_weights()

    def forward(self, img: torch.Tensor):
        states = self.backbone(img).last_hidden_state
        return states
    
