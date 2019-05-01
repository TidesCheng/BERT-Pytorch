import warnings


class DefaultConfig():

    # Device
    use_cuda = True

    # Path
    data_dir = 'data'
    corpus_file = 'bert_train_pairs.txt'
    vocab_path = 'vocab.dat'
    ckpt_path = 'checkpoints'

    ###### Hyper-parameters #############
    # Model
    max_vocab_size = None   # Determined by real vocab size from vocab.dat
    max_sen_len = 80
    n_layers = 12
    d_model = 512
    n_heads = 8
    n_k = 64
    n_v = 64
    n_seg = 3
    n_ff_hidden = d_model * 4
    max_mask_len = 15
    # Train
    batch_size = 16
    epochs = 8


def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
