import torch
import torch.nn as nn
from models import BERT
from config import opt
from get_bert_inputs import get_inputs
from models import BERTDataset, BERTCollate_fn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os

# Hyper-parameters for finetune
max_epochs = 1

#
class NextSenCLF(nn.Module):
    def __init__(self, d_model, bert):
        super(NextSenCLF, self).__init__()
        self.bert = bert
        self.clf = nn.Linear(d_model, 2)

    def forward(self, input_ids, seg_ids):
        output = self.bert(input_ids, seg_ids)
        output = self.clf(output)
        return output[:, 0]


def finetune(ckpt_file=None):

    if ckpt_file is None:
        files = os.listdir('checkpoints')
        if len(files):
            ckpt_file = files[-1]

    ckpt_path = os.path.join('checkpoints', ckpt_file)


    # Step 0: data and device
    inputs = get_inputs(data_dir=opt.data_dir, corpus_file=opt.corpus_file, vocab_file=opt.vocab_path)
    train_dataloader = DataLoader(dataset=BERTDataset(inputs, max_sen_len=opt.max_sen_len),
                                  shuffle=True,
                                  batch_size=opt.batch_size,
                                  collate_fn=BERTCollate_fn)
    use_cuda = True if opt.use_cuda and torch.cuda.is_available() else False
    if use_cuda:
        torch.cuda.empty_cache()
    device = torch.device('cuda' if use_cuda else 'cpu')

    writer = SummaryWriter(comment='finetune')

    # Step 1; model
    bert = BERT(n_layers=opt.n_layers,
                 d_model=opt.d_model,
                 vocab_size=opt.max_vocab_size,
                 max_len=opt.max_sen_len,
                 n_heads=opt.n_heads,
                 n_seg=opt.n_seg,
                 ff_hidden=opt.n_ff_hidden,
                 device=device).to(device)
    bert.load_state_dict(torch.load(ckpt_path))

    clf = NextSenCLF(d_model=opt.d_model, bert=bert).to(device)

    # Step: criterion and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(bert.parameters(), lr=1e-8)
    optimizer2 = torch.optim.Adam(clf.clf.parameters(), lr=0.001)

    # Step 3: training
    for epoch in range(max_epochs):
        epoch_loss = 0
        for i, batch_data in enumerate(train_dataloader, 1):

            input_ids, seg_ids, _, _, isnext = map(lambda x: x.to(device), batch_data)
            # Reset gradients and forward
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            clfout = clf(input_ids, seg_ids)

            # Compute loss
            clfout = clfout.view(-1, clfout.size(-1))
            isnext = isnext.view(-1,)
            loss = criterion(clfout, isnext)

            _, next_preds = torch.max(clfout, dim=-1)
            next_pred_acc = next_preds.eq(isnext).sum().item() / isnext.size(0)

            if i % 5 == 0:
                writer.add_scalar('clf_loss', loss.item(), i + epoch * len(train_dataloader))
                writer.add_scalar('clf_acc', next_pred_acc, i + epoch * len(train_dataloader))

                print('Epoch {}, Batch {}/{}, clf_loss={}, clf_acc={}'
                      .format(epoch + 1, i, len(train_dataloader), loss.item(), next_pred_acc))

            epoch_loss += loss.item()

            # Backward and update
            loss.backward()
            optimizer1.step()
            optimizer2.step()


if __name__=='__main__':
    import fire
    fire.Fire()


