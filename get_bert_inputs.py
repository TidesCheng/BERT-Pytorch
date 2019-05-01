import os
import pickle
from config import opt


def get_inputs(data_dir, corpus_file, vocab_file=None):

    corpus_path = os.path.join(data_dir, corpus_file)
    vocab_path = os.path.join(data_dir, vocab_file)
    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)

    opt.max_vocab_size = len(word2idx)
    idx2word = {i: w for w,i in word2idx.items()}
    input_ids, seg_ids, masked_tokens, masked_pos, isnext = [], [], [], [], []
    with open(corpus_path, 'r') as f:
        for line in f:
            s1, s2 = map(lambda x: x.strip(), line.strip().split('$$$')[:2])
            s1_ids = [word2idx[w] for w in s1 if w in word2idx]
            s2_ids = [word2idx[w] for w in s2 if w in word2idx]
            input_ids.append((s1_ids, s2_ids))

    inputs = {
        'word2idx': word2idx,
        'idx2word': idx2word,
        'input_ids': input_ids
    }
    return inputs


# if __name__=='__main__':
#     from config import opt
#     get_inputs(opt.data_dir, opt.corpus_file, opt.vocab_path)