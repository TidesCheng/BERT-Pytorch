import torch
import torch.nn as nn


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_seg, device):
        super(BERTEmbedding, self).__init__()
        print("vocab size " + str(vocab_size))
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.seg_emb = nn.Embedding(n_seg, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x).to(self.device)    # (seq_len,) -> (batch_size, seq_len)

        embedding = self.token_emb(x) + self.pos_emb(pos) + self.pos_emb(seg)
        return self.layer_norm(embedding)





