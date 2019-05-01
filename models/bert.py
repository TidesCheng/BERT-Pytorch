import torch
import torch.nn as nn
from .embeddings import BERTEmbedding
from .sublayers import EncoderLayer, gelu


def get_attn_pad_mask(seq_q, seq_k):
    bz, len_q = seq_q.size()
    bz, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(bz, len_q, len_k)


class BERT(nn.Module):
    def __init__(self, n_layers, d_model, vocab_size, max_len, n_seg, ff_hidden, n_heads, device):
        super(BERT, self).__init__()
        self.embedding = BERTEmbedding(d_model=d_model, vocab_size=vocab_size, max_len=max_len, n_seg=n_seg, device=device)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_heads=n_heads, ff_hidden=ff_hidden)
                                         for _ in range(n_layers)])

    def forward(self, input_ids, seg_ids):
        # Embedding
        output = self.embedding(input_ids, seg_ids)
        # Mask
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        # Encoder layers
        for encoder in self.enc_layers:
            output, enc_self_attn = encoder(output, enc_self_attn_mask)
        return output


class MaskedLM(nn.Module):
    def __init__(self, d_model, vocab_size, bert):
        super(MaskedLM, self).__init__()
        # self.linear = nn.Linear(d_model, d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
        # Decoder part share embedding with token embedding
        self.decoder.weight = bert.embedding.token_emb.weight
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
        self.act = gelu

    def forward(self, bertout, masked_pos):
        masked_pos = masked_pos.unsqueeze(2).expand(-1, -1, bertout.size(-1))
        h_masked = torch.gather(bertout, 1, masked_pos)
        # h_masked = self.act(self.linear(h_masked))
        logit_lm = self.decoder(h_masked)

        return logit_lm


class NextPred(nn.Module):
    def __init__(self, d_model):
        super(NextPred, self).__init__()
        self.clf = nn.Linear(d_model, 2)
        self.act = nn.Sigmoid()

    def forward(self, bertout):
        return self.clf(self.act(bertout[:, 0]))


#
# class MaskedLM(nn.Module):
#     def __init__(self, d_model, vocab_size, bert):
#         super(MaskedLM, self).__init__()
#         self.linear = nn.Linear(d_model, d_model)
#         self.decoder = nn.Linear(d_model, vocab_size)
#         # Decoder part share embedding with token embedding
#         self.decoder.weight = bert.embedding.token_emb.weight
#         self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
#         self.act = gelu
#
#     def forward(self, bertout, masked_pos):
#         masked_pos = masked_pos.unsqueeze(2).expand(-1, -1, bertout.size(-1))
#         h_masked = torch.gather(bertout, 1, masked_pos)
#         h_masked = self.act(self.linear(h_masked))
#         logit_lm = self.decoder(h_masked)
#
#         return logit_lm


# class NextPred(nn.Module):
#     def __init__(self, d_model):
#         super(NextPred, self).__init__()
#         self.clf = nn.Linear(d_model, 2)
#         self.act = nn.Sigmoid()
#
#     def forward(self, bertout):
#         return self.clf(self.act(bertout[:, 0]))

# class BERT(nn.Module):
#     def __init__(self, n_layers, d_model, vocab_size, max_len, n_seg, ff_hidden, n_heads, device):
#         super(BERT, self).__init__()
#         self.embedding = BERTEmbedding(d_model=d_model, vocab_size=vocab_size, max_len=max_len, n_seg=n_seg, device=device)
#         self.enc_layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_heads=n_heads, ff_hidden=ff_hidden)
#                                          for _ in range(n_layers)])
#         self.fc = nn.Linear(d_model, d_model)
#         self.act1 = nn.Sigmoid()
#         self.linear = nn.Linear(d_model, d_model)
#         self.act2 = gelu
#
#         self.clf = nn.Linear(d_model, 2)
#
#         self.norm = nn.LayerNorm(d_model)
#
#         # Decoder part share embedding with token embedding
#         self.decoder = nn.Linear(d_model, vocab_size)
#         self.decoder.weight = self.embedding.token_emb.weight
#         self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
#
#     def forward(self, input_ids, seg_ids, masked_pos):
#         # Embedding
#         output = self.embedding(input_ids, seg_ids)
#         # Mask
#         enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
#         # Encoder layers
#         for encoder in self.enc_layers:
#             output, enc_self_attn = encoder(output, enc_self_attn_mask)
#         # For next sentence prediction
#         logits_clsf = self.clf(self.act1(output[:, 0]))     # bz * 2
#         # For masked words prediction
#         masked_pos = masked_pos.unsqueeze(2).expand(-1, -1, output.size(-1))
#         h_masked = torch.gather(output, 1, masked_pos)
#         h_masked = self.norm(self.act2(self.linear(h_masked)))
#         logits_lm = self.decoder(h_masked) + self.decoder_bias  # bz * len_mask * vocab
#
#         return logits_clsf, logits_lm