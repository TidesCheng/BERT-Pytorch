import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale=True):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        scores = torch.bmm(q, k.permute(0, 2, 1))

        if self.scale is not None:
            scores = scores / np.sqrt(k.size(2))

        if mask is not None:
            scores = scores.masked_fill(mask, 1e-9)

        attn = F.softmax(scores, dim=2)

        context = torch.bmm(attn, v)

        return context, attn

