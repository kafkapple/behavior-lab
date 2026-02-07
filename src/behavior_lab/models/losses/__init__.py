"""Loss functions: label smoothing, MMD."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -logprobs.mean(dim=-1)
        return (confidence * nll + self.smoothing * smooth).mean()


def get_mmd_loss(z, z_prior, y, num_cls):
    """MMD loss: match class-conditional z to prior."""
    y_valid = [i in y for i in range(num_cls)]
    z_mean = torch.stack([
        z[y == i].mean(dim=0) if i in y else torch.zeros_like(z[0])
        for i in range(num_cls)], dim=0)
    l2_z_mean = LA.norm(z.mean(dim=0), ord=2)
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return mmd_loss, l2_z_mean
