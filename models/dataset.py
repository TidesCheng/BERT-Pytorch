import torch
from torch.utils.data import Dataset
import random
from config import opt

class BERTDataset(Dataset):
    def __init__(self, inputs, max_sen_len=None):
        print("max_sen_len " + str(max_sen_len))
        super(BERTDataset, self).__init__()

        self.idx2word = inputs['idx2word']
        self.word2idx = inputs['word2idx']
        if max_sen_len is None:
            self.samples = inputs
        else:
            samples = inputs['input_ids']
            filtered_samples = []
            for sample in samples:
                if len(sample[0]) + len(sample[1]) <= max_sen_len - 3:
                    filtered_samples.append(sample)
            self.samples = filtered_samples
        self.cls = self.word2idx['[CLS]']
        self.sep = self.word2idx['[SEP]']
        self.mask = self.word2idx['[MASK]']
        self.max_sen_len = max_sen_len

    def __getitem__(self, item):
        s1, s2 = self.samples[item]
        isnext = 1

        # Randomly replace second sentence
        prb = random.random()
        if prb < 0.5:
            cand_s2 = self.samples[random.randint(0, len(self.samples)-1)][0]
            while len(cand_s2) > len(s2):
                cand_s2 = self.samples[random.randint(0, len(self.samples) - 1)][0]
            s2 = cand_s2
            isnext = 0

        # Adding flags to form index sequence
        input_ids = [self.cls] + s1 + [self.sep] + s2 + [self.sep]
        seg_ids = [1] * (len(s1) + 2) + [2] * (len(s2) + 1)

        # Randomly mask some words
        cand_masked_pos = [i for i,token in enumerate(input_ids) if token != self.cls and token != self.sep]
        masked_pos, masked_token = [], []
        for pos in cand_masked_pos:
            prb = random.random()
            if prb < 0.15:      # Mask word with prb=0.15
                masked_pos.append(pos)
                masked_token.append(input_ids[pos])
                prb /= 0.15
                if prb < 0.8:   # Replace input as '[MASK]' with prb=0.8
                    input_ids[pos] = self.mask
                elif prb > 0.9: # Replace input as random word with prb=0.1
                    input_ids[pos] = random.randint(0, len(self.word2idx)-1)
                # Remain unchanged with prb=0.1

        # Zero padding
        n_pad = self.max_sen_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        seg_ids.extend([0] * n_pad)

        # The training sample consists of 5 parts
        train_sample = [input_ids, seg_ids, masked_pos, masked_token, isnext]
        return train_sample

    def __len__(self):
        return len(self.samples)


def BERTCollate_fn(train_samples):

    input_ids, seg_ids, masked_pos, masked_token, isnext = list(zip(*train_samples))

    # max_pred_len is determined by the batch
    max_mask_len = max(len(mp) for mp in masked_pos)
    # max_mask_len = opt.max_mask_len
    for mp in masked_pos:
        mp.extend([0] * (max_mask_len - len(mp)))
        # mp = mp[:max_mask_len]

    for mt in masked_token:
        mt.extend([0] * (max_mask_len - len(mt)))
        # mt = mt[:max_mask_len]


    input_ids = torch.LongTensor(input_ids)
    seg_ids = torch.LongTensor(seg_ids)
    masked_pos = torch.LongTensor(masked_pos)
    masked_token = torch.LongTensor(masked_token)
    isnext = torch.LongTensor(isnext)

    # input_ids, seg_ids, masked_pos, masked_token, isnext = tuple(map(torch.LongTensor, (input_ids, seg_ids, masked_pos, masked_token, isnext)))

    # return (*input_ids, *seg_ids, *masked_pos, *masked_token, *isnext)
    return (input_ids, seg_ids, masked_pos, masked_token, isnext)