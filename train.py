from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import json
from konlpy.tag import Mecab

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from evaluate import evaluate, decoding_from_result
from metric import acc
from model.net import Transformer
from model.optim import GradualWarmupScheduler

from data_utils.utils import Config, CheckpointManager, SummaryManager
from data_utils.chatbot_dataset import ChatbotDataset
from data_utils.vocab_tokenizer import Tokenizer, Vocabulary, keras_pad_fn, mecab_token_pos_flat_fn

np.set_printoptions(suppress=False)
np.set_printoptions(threshold=sys.maxsize)



def main(parser):
    # Config
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    data_config = Config(json_path=data_dir / 'config.json')
    model_config = Config(json_path=model_dir / 'config.json')

    # Vocab & Tokenizer
    with open(data_config.token2idx_vocab, mode='rb') as io:
        token2idx_vocab = json.load(io)
        print("token2idx_vocab: ", token2idx_vocab)
    vocab = Vocabulary(token2idx = token2idx_vocab)
    tokenizer = Tokenizer(vocab=vocab, split_fn=mecab_token_pos_flat_fn, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)
    model_config.vocab_size = len(vocab.token2idx)

    # Model & Model Params
    model = Transformer(config=model_config, vocab=vocab)

    # Train & Val Datasets
    tr_ds = ChatbotDataset(data_config.train, tokenizer.list_of_string_to_arr_of_pad_token_ids)
    tr_dl = DataLoader(tr_ds, batch_size=model_config.batch_size, shuffle=True, num_workers=4, drop_last=False)

    val_ds = ChatbotDataset(data_config.validation, tokenizer.list_of_string_to_arr_of_pad_token_ids)
    val_dl = DataLoader(val_ds, batch_size=model_config.batch_size, shuffle=True, num_workers=4, drop_last=False)

    # loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID) # nn.NLLLoss()

    # optim
    opt = optim.Adam(params=model.parameters(), lr=model_config.learning_rate) # torch.optim.SGD(params=model.parameters(), lr=model_config.learning_rate)
    # scheduler = ReduceLROnPlateau(opt, patience=5)  # Check
    scheduler = GradualWarmupScheduler(opt, multiplier=8, total_epoch=model_config.epochs)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # save
    # writer = SummaryWriter('{}/runs'.format(model_dir))
    checkpoint_manager = CheckpointManager(model_dir)
    summary_manager = SummaryManager(model_dir)
    best_val_loss = 1e+10
    best_train_acc = 0

    # load
    if (model_dir / 'best.tar').exists():
        print("pretrained model exists")
        checkpoint = checkpoint_manager.load_checkpoint('best.tar')
        model.load_state_dict(checkpoint['model_state_dict'])

    # Train
    for epoch in tqdm(range(model_config.epochs), desc='epoch', total=model_config.epochs):
        scheduler.step(epoch)
        print("epoch : {}, lr: {}".format(epoch, opt.param_groups[0]['lr']))
        tr_loss = 0
        tr_acc = 0
        model.train()

        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            opt.zero_grad()

            enc_input, dec_input, dec_output = map(lambda elm: elm.to(device), mb)
            y_pred = model(enc_input, dec_input)
            y_pred_copy = y_pred.detach()
            dec_output_copy = dec_output.detach()

            # loss 계산을 위해 shape 변경
            y_pred = y_pred.reshape(-1, y_pred.size(-1))
            dec_output = dec_output.view(-1).long()

            # padding 제외한 value index 추출
            real_value_index = [dec_output != 0]

            # padding은 loss 계산시 제외
            mb_loss = loss_fn(y_pred[real_value_index], dec_output[real_value_index]) # Input: (N, C) Target: (N)
            mb_loss.backward()
            opt.step()

            with torch.no_grad():
                mb_acc = acc(y_pred, dec_output)

            tr_loss += mb_loss.item()
            tr_acc = mb_acc.item()
            tr_loss_avg =  tr_loss / (step + 1)
            tr_summary = {'loss': tr_loss_avg, 'acc': tr_acc}
            total_step = epoch * len(tr_dl) + step

            # Eval
            if total_step % model_config.summary_step ==0 and total_step != 0:
                print("train: ")
                decoding_from_result(enc_input, y_pred_copy, dec_output_copy, tokenizer)

                model.eval()
                print("eval: ")
                val_summary = evaluate(model, val_dl, {'loss': loss_fn, 'acc':acc}, device, tokenizer)
                val_loss = val_summary['loss']

                # writer.add_scalars('loss', {'train': tr_loss_avg,
                #                             'val': val_loss}, epoch * len(tr_dl) + step)

                tqdm.write('epoch : {}, step : {}, '
                           'tr_loss: {:.3f}, val_loss: {:.3f}, tr_acc: {:.2%}, val_acc: {:.2%}'.format(epoch + 1, total_step,
                                                                            tr_summary['loss'],
                                                                            val_summary['loss'], tr_summary['acc'],
                                                                            val_summary['acc']))

                val_loss = val_summary['loss']
                # is_best = val_loss < best_val_loss # loss 기준
                is_best = tr_acc > best_train_acc # acc 기준 (원래는 train_acc가 아니라 val_acc로 해야)

                # Save
                if is_best:
                    print("[Best model Save] train_acc: {}, train_loss: {}, val_loss: {}".format(tr_summary['acc'], tr_summary['loss'], val_loss))
                    # CPU에서도 동작 가능하도록 자료형 바꾼 뒤 저장
                    state = {'epoch': epoch + 1,
                             'model_state_dict': model.to(torch.device('cpu')).state_dict(),
                             'opt_state_dict': opt.state_dict()}
                    summary = {'train': tr_summary, 'validation': val_summary}

                    summary_manager.update(summary)
                    summary_manager.save('summary.json')
                    checkpoint_manager.save_checkpoint(state, 'best.tar')

                    best_val_loss = val_loss

                model.to(device)
                model.train()
            else:
                if step % 50 == 0:
                    print('epoch : {}, step : {}, tr_loss: {:.3f}, tr_acc: {:.2%}'.format(epoch + 1, total_step, tr_summary['loss'], tr_summary['acc']))





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data_in', help="Directory containing config.json of data")
    parser.add_argument('--model_dir', default='experiments/base_model',
                        help="Directory containing config.json of model")

    main(parser)