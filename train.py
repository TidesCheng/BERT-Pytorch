import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import opt
from models import BERT, BERTDataset, BERTCollate_fn, MaskedLM, NextPred

from tensorboardX import SummaryWriter
from get_bert_inputs import get_inputs
import os
from datetime import datetime as dt

def train(**kwargs):

    # The received parameters will be used to update configuration dict
    opt.parse(kwargs)

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
    writer = SummaryWriter()

    # Step 1: model
    bert = BERT(n_layers=opt.n_layers,
               d_model=opt.d_model,
               vocab_size=opt.max_vocab_size,
               max_len=opt.max_sen_len,
               n_heads=opt.n_heads,
               n_seg=opt.n_seg,
               ff_hidden=opt.n_ff_hidden,
               device=device).to(device)
    masked_lm = MaskedLM(d_model=opt.d_model, vocab_size=opt.max_vocab_size, bert=bert).to(device)
    next_pred = NextPred(d_model=opt.d_model).to(device)

    # Write model
    dummy_input_ids = torch.zeros((opt.batch_size, opt.max_sen_len)).long().to(device)
    dummy_seg_ids = torch.zeros((opt.batch_size, opt.max_sen_len)).long().to(device)

    writer.add_graph(bert, (dummy_input_ids, dummy_seg_ids), False)

    # dummy_bertout = torch.zeros((opt.batch_size, opt.max_sen_len, opt.d_model)).long().to(device)
    # dummy_masked_pos = torch.zeros((opt.batch_size, opt.max_mask_len)).long().to(device)
    #
    # writer.add_graph(masked_lm, (dummy_bertout, dummy_masked_pos), True)
    # writer.add_graph(next_pred, (dummy_bertout), True)

    # Step 2: criterion and optimizer
    criterion = nn.CrossEntropyLoss()

    num_paras = sum(p.numel() for model in (bert, masked_lm, next_pred) for p in model.parameters() if p.requires_grad)
    paras = list(bert.parameters()) + list(masked_lm.parameters()) + list(next_pred.parameters())
    print("Total number of parameters is {}".format(num_paras))
    optimizer = torch.optim.Adam(paras, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

    # Step 3: train
    print("Start training ...")
    for epoch in range(opt.epochs):
        epoch_loss = 0
        for i, batch_data in enumerate(train_dataloader, 1):

            input_ids, seg_ids, masked_pos, masked_token, isnext = map(lambda x: x.to(device), batch_data)
            # Reset gradients and forward
            optimizer.zero_grad()
            bertout = bert(input_ids, seg_ids)
            logits_lm = masked_lm(bertout, masked_pos)
            logits_clsf = next_pred(bertout)

            # Compute loss
            logits_lm = logits_lm.view(-1, logits_lm.size(-1))  # (bz * len_mask, vocab)
            masked_token = masked_token.view(-1,)   # (bz * len_mask, )
            logits_clsf = logits_clsf.view(-1, logits_clsf.size(-1))    # (bz, )
            isnext.view(-1,)            # (bz, )

            loss_lm = criterion(logits_lm, masked_token)
            loss_clsf = criterion(logits_clsf, isnext)
            loss = loss_lm + loss_clsf

            _, mask_preds = torch.max(logits_lm, dim=-1)
            _, next_preds = torch.max(logits_clsf, dim=-1)
            mask_pred_acc = mask_preds.eq(masked_token).sum().item() / masked_token.size(0)
            next_pred_acc = next_preds.eq(isnext).sum().item() / isnext.size(0)

            if i % 20 == 0:
                writer.add_scalar('loss_lm', loss_lm.item(), i + epoch * len(train_dataloader))
                writer.add_scalar('loss_clsf', loss_clsf.item(), i + epoch * len(train_dataloader))
                writer.add_scalar('lm_acc', mask_pred_acc, i + epoch * len(train_dataloader))
                writer.add_scalar('next_acc', next_pred_acc, i + epoch * len(train_dataloader))
                print('Epoch {}, Batch {}/{}, loss_lm={}, loss_next={}, lm_acc={}, next_acc={}'
                      .format(epoch+1, i, len(train_dataloader), loss_lm.item(), loss_clsf.item(), mask_pred_acc, next_pred_acc))

            epoch_loss += loss.item()

            # Backward and update
            loss.backward()
            optimizer.step()

        if (1 + epoch) % 1 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(epoch_loss))

    print('finished train')

    # Step 4: Save model
    ckpt_file_name = dt.strftime(dt.now(), '%Y-%m-%d %H: %M: %S.ckpt')
    save_path = os.path.join(opt.ckpt_path, ckpt_file_name)
    torch.save(bert.state_dict(), save_path)


if __name__ == '__main__':
    import fire
    fire.Fire()
    # train()




