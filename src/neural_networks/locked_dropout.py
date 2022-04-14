import torch
import torch.nn as nn


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout: float = 0.5):
        if not self.training or not dropout:    # Model is eval mode
            return x
        mask = torch.bernoulli(x, 1 - dropout)
        mask = mask / (1 - dropout)
        mask.requires_grad = False
        return mask * x
