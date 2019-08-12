from __future__ import absolute_import, division, print_function, unicode_literals
import os
import torch
from sklearn.model_selection import train_test_split
from tensorflow import keras
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pprint import pprint
from typing import Tuple, Callable, List # https://m.blog.naver.com/PostView.nhn?blogId=passion053&logNo=221070020739&proxyReferer=https%3A%2F%2Fwww.google.com%2F

from tqdm import tqdm


class ChatbotDataset(Dataset):
    def __init__(self, filepath: str, transform_fn: Callable[[str], List[int]]) -> None:
        """

        :param filepath:
        :param transform_fn:
        """
        question, answer = self.load_data_from_txt(filepath)

        self._corpus = question
        self._label = answer
        self._transform = transform_fn

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_input = torch.tensor(self._transform([self._corpus[idx].lower()]))
        dec_input, dec_output = torch.tensor(self._transform([self._label[idx].lower()], add_start_end_token=True))
        return enc_input[0], dec_input[0], dec_output[0]

    def load_data_from_txt(self, data_path):
        with open(data_path, mode='r', encoding='utf-8') as io:
            lines = io.readlines()
            question = []
            answer = []
            for line in lines:
                if line == "":
                    continue
                question_item, answer_item = line.split('\t')
                question.append(question_item)
                answer.append(answer_item)
        return question, answer

    def load_data(self, data_path, train_val_split=True):
        """
        code from "https://github.com/changwookjun/Transformer/blob/25d9472155cb0788d11dbfe274526690915fe95e/data.py#L27"
        :param data_path:
        :return: train_input, train_label, eval_input, eval_label
        """
        # 판다스를 통해서 데이터를 불러온다.
        data_df = pd.read_csv(data_path, header=0)
        # 질문과 답변 열을 가져와 question과 answer에 넣는다.
        question, answer = list(data_df['Q']), list(data_df['A'])
        # skleran에서 지원하는 함수를 통해서 학습 셋과
        # 테스트 셋을 나눈다.
        if train_val_split:
            train_input, eval_input, train_label, eval_label = train_test_split(question, answer, test_size=0.33,
                                                                            random_state=42)
            return train_input, train_label, eval_input, eval_label
        else:
            return question, answer
        # 그 값을 리턴한다.