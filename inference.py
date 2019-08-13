from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import sys
import numpy as np
from pathlib import Path

import json
from konlpy.tag import Mecab

import torch
from evaluate import decoding_from_result
from model.net import Transformer

from data_utils.utils import Config, CheckpointManager, SummaryManager
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

    # Model
    model = Transformer(config=model_config, vocab=vocab)
    checkpoint_manager = CheckpointManager(model_dir) # experiments/base_model
    checkpoint = checkpoint_manager.load_checkpoint('best.tar')
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    while(True):
        input_text = input("문장을 입력하세요: ")
        enc_input = torch.tensor(tokenizer.list_of_string_to_arr_of_pad_token_ids([input_text]))
        dec_input = torch.tensor([[vocab.token2idx[vocab.START_TOKEN]]])

        for i in range(model_config.maxlen):
            y_pred = model(enc_input.to(device), dec_input.to(device))
            y_pred_ids = y_pred.max(dim=-1)[1]
            if (y_pred_ids[0,-1] == vocab.token2idx[vocab.END_TOKEN]).to(torch.device('cpu')).numpy():
                decoding_from_result(enc_input=enc_input, y_pred=y_pred, tokenizer=tokenizer)
                break

            # decoding_from_result(enc_input, y_pred, tokenizer)
            dec_input = torch.cat([dec_input.to(torch.device('cpu')), y_pred_ids[0,-1].unsqueeze(0).unsqueeze(0).to(torch.device('cpu'))], dim=-1)

            if i == model_config.maxlen - 1:
                decoding_from_result(enc_input=enc_input, y_pred=y_pred, tokenizer=tokenizer)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data_in', help="Directory containing config.json of data")
    parser.add_argument('--model_dir', default='./experiments/base_model',
                        help="Directory containing config.json of model")

    main(parser)