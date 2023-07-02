# Code from https://github.com/nihalsid/stylegan2-ada-lightning

import torch
import numpy as np
from torch import nn


def compute_gradient_penalty(x, d):
    gradients = torch.autograd.grad(outputs=[d.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
    r1_penalty = gradients.square().sum([1, 2, 3]).mean()
    return r1_penalty / 2


def compute_gradient_penalty_patch(x, d, idx):
    gradients = torch.autograd.grad(outputs=[d.sum()], inputs=[x], create_graph=True, only_inputs=True)[0][:, :, idx, :, :]
    r1_penalty = gradients.square().sum([1, 2, 3]).mean()
    return r1_penalty / 2


class PathLengthPenalty(nn.Module):

    def __init__(self, pl_decay, pl_batch_shrink):
        super().__init__()
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_mean = nn.Parameter(torch.zeros([1]), requires_grad=False)

    def forward(self, fake, w):
        pl_noise = torch.randn_like(fake) / np.sqrt(fake.shape[2] * fake.shape[3])
        pl_grads = torch.autograd.grad(outputs=[(fake * pl_noise).sum()], inputs=[w], create_graph=True, only_inputs=True)[0]
        pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        return pl_penalty.mean()
