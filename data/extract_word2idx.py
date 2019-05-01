import os
import collections
import pickle


def extract(corpus_path='bert_train_pairs.txt', max_vocab_size=200000):

    word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

    word_counter = collections.defaultdict(int)
    with open(corpus_path, 'r') as f:
        for line in f:
            s1, s2 = map(lambda x: x.strip(), line.strip().split('$$$')[:2])
            for w in s1.split():
                word_counter[w] += 1
            for w in s2.split():
                word_counter[w] += 1

    if max_vocab_size is None:
        max_vocab_size = len(word_counter) + 4
    for w, f in sorted(list(word_counter.items())[:max_vocab_size], key=lambda x: -x[1]):
        word2idx[w] = len(word2idx)

    vocab_path = 'vocab.dat'
    with open(vocab_path, 'wb') as f:
        pickle.dump(word2idx, f)

    print("The size of vocab is {}".format(len(word2idx)))


if __name__=='__main__':
    import fire
    fire.Fire()



