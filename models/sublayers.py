import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .attentions import ScaledDotProductAttention


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, ff_hidden):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(n_head=n_heads, d_model=d_model, dk=d_model // n_heads, dv=d_model // n_heads)
        self.pos_ff = PositionsiseFeedForward(d_model=d_model, d_hid=ff_hidden)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_ouputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_ouputs = self.pos_ff(enc_ouputs)

        return enc_ouputs, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dk, dv):
        # (len_q * Dq) * (Dk, len_k) * (len_v * Dv)
        # so it is assumed that Dq == Dk (dq == dk) and  len_k == len_v

        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.dk = dk
        self.dv = dv

        self.attn = ScaledDotProductAttention(dk)
        self.linear_q = nn.Linear(d_model, n_head * dk)
        self.linear_k = nn.Linear(d_model, n_head * dk)
        self.linear_v = nn.Linear(d_model, n_head * dv)
        nn.init.normal_(self.linear_q.weight, mean=0, std=np.sqrt(2.0 / (d_model + dk)))
        nn.init.normal_(self.linear_k.weight, mean=0, std=np.sqrt(2.0 / (d_model + dk)))
        nn.init.normal_(self.linear_v.weight, mean=0, std=np.sqrt(2.0 / (d_model + dv)))

        self.fc = nn.Linear(n_head * dv, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(n_head * dv)

    def forward(self, Q, K, V, mask=None):

        residual = Q

        bz = Q.size(0)
        q = self.linear_q(Q).view(bz, -1, self.n_head, self.dk).permute(0, 2, 1, 3).contiguous().view(bz * self.n_head, -1, self.dk)
        k = self.linear_k(Q).view(bz, -1, self.n_head, self.dk).permute(0, 2, 1, 3).contiguous().view(bz * self.n_head, -1, self.dk)
        v = self.linear_v(Q).view(bz, -1, self.n_head, self.dv).permute(0, 2, 1, 3).contiguous().view(bz * self.n_head, -1, self.dv)

        mask = mask.repeat(self.n_head, 1, 1)
        context, attn = self.attn(q, k, v, mask)

        context = context.view(bz, self.n_head, -1, self.dv).permute(0, 2, 1, 3).contiguous().view(bz, -1, self.n_head * self.dv)

        context = self.fc(context)

        context = self.layer_norm(context + residual)

        return context, attn


class PositionsiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hid):
        super(PositionsiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_hid)
        self.w2 = nn.Linear(d_hid, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.w2(gelu(self.w1(x)))
        x = self.layer_norm(x + residual)
        return x

# def gelu(x):
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
