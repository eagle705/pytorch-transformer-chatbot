from __future__ import absolute_import, division, print_function, unicode_literals
import os
from tensorflow import keras
import numpy as np
from pprint import pprint
from konlpy.tag import Mecab

import sys
import pickle
import os
import codecs
import argparse
from collections import Counter
from threading import Thread
from tqdm import tqdm

mecab = Mecab()

class Vocabulary(object):
    """Vocab Class"""

    def __init__(self, token2idx=None):

        self.token2idx = {}
        self.idx2token = {}
        self.idx = 0

        self.PAD = "<pad>";
        self.START_TOKEN = "<s>";
        self.END_TOKEN = "</s>"
        self.UNK = "<unk>";
        self.CLS = "[CLS]";
        self.MASK = "[MASK]"
        self.SEP = "[SEP]";
        self.SEG_A = "[SEG_A]";
        self.SEG_B = "[SEG_B]"
        self.NUM = "<num>";


        self.special_tokens = [self.PAD,
                                self.START_TOKEN,
                                self.END_TOKEN,
                                self.UNK,
                                self.CLS,
                                self.MASK,
                                self.SEP,
                                self.SEG_A,
                                self.SEG_B,
                                self.NUM]
        self.init_vocab()


        if token2idx is not None:
            self.token2idx = token2idx
            self.idx2token = {v:k for k,v in token2idx.items()}
            self.idx = len(token2idx) - 1

        self.PAD_ID = self.transform_token2idx(self.PAD)

    def init_vocab(self):
        for special_token in self.special_tokens:
            self.add_token(special_token)

    def to_indices(self, tokens):
        return [self.transform_token2idx(X_token) for X_token in tokens]

    def add_token(self, token):
        if not token in self.token2idx:
            self.token2idx[token] = self.idx
            self.idx2token[self.idx] = token
            self.idx += 1

    def transform_token2idx(self, token, show_oov=False):
        try:
            return self.token2idx[token]
        except:
            if show_oov is True:
                print("key error: "+ str(token))
            token = self.UNK
            return self.token2idx[token]

    def transform_idx2token(self, idx):
        try:
            return self.idx2token[idx]
        except:
            print("key error: " + str(idx))
            idx = self.token2idx[self.UNK]
            return self.idx2token[idx]

    def __len__(self):
        return len(self.token2idx)

    def build_vocab(self, list_of_str, threshold=1, vocab_save_path="./data_in/token_vocab.json", split_fn=Mecab().morphs):
        """Build a token vocab"""

        def do_concurrent_tagging(start, end, text_list, counter):
            for i, text in enumerate(text_list[start:end]):
                text = text.strip()
                text = text.lower()

                try:
                    tokens_ko = split_fn(text)
                    # tokens_ko = [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]
                    counter.update(tokens_ko)

                    if i % 1000 == 0:
                        print("[%d/%d (total: %d)] Tokenized input text." % (
                            start + i, start + len(text_list[start:end]), len(text_list)))

                except Exception as e:  # OOM, Parsing Error
                    print(e)
                    continue

        counter = Counter()

        num_thread = 4
        thread_list = []
        num_list_of_str = len(list_of_str)
        for i in range(num_thread):
            thread_list.append(Thread(target=do_concurrent_tagging, args=(
                int(i * num_list_of_str / num_thread), int((i + 1) * num_list_of_str / num_thread), list_of_str, counter)))

        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()

        # vocab_report
        print(counter.most_common(10))  # print most common tokens
        tokens = [token for token, cnt in counter.items() if cnt >= threshold]

        for i, token in enumerate(tokens):
            self.add_token(str(token))

        print("len(self.token2idx): ", len(self.token2idx))

        import json
        with open(vocab_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=4)

        return self.token2idx


def mecab_token_pos_flat_fn(string):
    tokens_ko = mecab.pos(string)
    return [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]

def mecab_token_pos_sep_fn(string):
    tokens_ko = mecab.pos(string)
    list_of_token = []
    list_of_pos = []
    for token, pos in tokens_ko:
        list_of_token.append(token)
        list_of_pos.append(pos)
    return list_of_token, list_of_pos

def mecab_token_fn(string):
    tokens_ko = mecab.pos(string)
    return [str(pos[0]) for pos in tokens_ko]

def mecab_pos_fn(string):
    tokens_ko = mecab.pos(string)
    return [str(pos[0]) for pos in tokens_ko]

def keras_pad_fn(token_ids_batch, maxlen, pad_id=0, padding='post', truncating='post'):
    padded_token_ids_batch = keras.preprocessing.sequence.pad_sequences(token_ids_batch,
                                                                    value=pad_id,#vocab.transform_token2idx(PAD),
                                                                    padding=padding,
                                                                    truncating=truncating,
                                                                    maxlen=maxlen)
    return np.array(padded_token_ids_batch)


class Tokenizer:
    """ Tokenizer class"""
    def __init__(self, vocab, split_fn, pad_fn, maxlen):
        self._vocab = vocab
        self._split = split_fn
        self._pad = pad_fn
        self._maxlen = maxlen

    # def split(self, string: str) -> list[str]:
    def split(self, string):
        tokens = self._split(string)
        return tokens

    # def transform(self, list_of_tokens: list[str]) -> list[int]:
    def transform(self, tokens):
        indices = self._vocab.to_indices(tokens)
        pad_indices = self._pad(indices, pad_id=0, maxlen=self._maxlen) if self._pad else indices
        return pad_indices

    # def split_and_transform(self, string: str) -> list[int]:
    def split_and_transform(self, string):
        return self.transform(self.split(string))

    @property
    def vocab(self):
        return self._vocab

    # def list_of_string_to_two_list_of_tokens_poses(self, X_str_batch):
    #     X_token_batch = []
    #     X_pos_batch = []
    #     for X_str in X_str_batch:
    #         x_token, x_pos = self._split(X_str)
    #         X_token_batch.append(x_token)
    #         X_pos_batch.append(x_pos)
    #     return X_token_batch, X_pos_batch

    def list_of_tokens_to_list_of_token_ids(self, X_token_batch):
        X_ids_batch = []
        for X_tokens in X_token_batch:
            X_ids_batch.append([self._vocab.transform_token2idx(X_token) for X_token in X_tokens])
        return X_ids_batch

    def list_of_string_to_list_of_tokens(self, X_str_batch):
        X_token_batch = [self._split(X_str) for X_str in X_str_batch]
        return X_token_batch

    def list_of_tokens_to_list_of_token_ids(self, X_token_batch):
        X_ids_batch = []
        for X_tokens in X_token_batch:
            X_ids_batch.append([self._vocab.transform_token2idx(X_token) for X_token in X_tokens])
        return X_ids_batch

    def list_of_string_to_list_token_ids(self, X_str_batch):
        X_token_batch = self.list_of_string_to_list_of_tokens(X_str_batch)
        X_ids_batch = self.list_of_tokens_to_list_of_token_ids(X_token_batch)

        return X_ids_batch

    def list_of_string_to_arr_of_pad_token_ids(self, X_str_batch, add_start_end_token=False):
        X_token_batch = self.list_of_string_to_list_of_tokens(X_str_batch)
        if add_start_end_token is True:
            return self.add_start_end_token_with_pad(X_token_batch)
        else:
            X_ids_batch = self.list_of_tokens_to_list_of_token_ids(X_token_batch)
            pad_X_ids_batch = self._pad(X_ids_batch, pad_id=self._vocab.PAD_ID, maxlen=self._maxlen)
        return pad_X_ids_batch

    def add_start_end_token_with_pad(self, X_token_batch):
        dec_input_token_batch = [[self._vocab.START_TOKEN] + X_token for X_token in X_token_batch]
        dec_output_token_batch = [X_token + [self._vocab.END_TOKEN] for X_token in X_token_batch]

        dec_input_token_batch = self.list_of_tokens_to_list_of_token_ids(dec_input_token_batch)
        pad_dec_input_ids_batch = self._pad(dec_input_token_batch, pad_id=self._vocab.PAD_ID, maxlen=self._maxlen)

        dec_output_ids_batch = self.list_of_tokens_to_list_of_token_ids(dec_output_token_batch)
        pad_dec_output_ids_batch = self._pad(dec_output_ids_batch, pad_id=self._vocab.PAD_ID, maxlen=self._maxlen)
        return pad_dec_input_ids_batch, pad_dec_output_ids_batch

    def decode_token_ids(self, token_ids_batch):
        token_token_batch = []
        for token_ids in token_ids_batch:
            token_token = [self._vocab.transform_idx2token(token_id) for token_id in token_ids]
            # token_token = [self._vocab[token_id] for token_id in token_ids]
            token_token_batch.append(token_token)
        return token_token_batch

        return ' '.join([reverse_token_index.get(i, '?') for i in text])



def main():
    print(mecab_token_pos_sep_fn("안녕하세요"))

if __name__ == '__main__':
    main()