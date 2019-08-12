"""
ref: https://github.com/aisolab/nlp_implementation/blob/master/A_Structured_Self-attentive_Sentence_Embedding_sts/build_vocab.py

"""
import pickle
import pandas as pd

import itertools
from pathlib import Path
from sklearn.model_selection import train_test_split
from data_utils.vocab_tokenizer import Vocabulary, mecab_token_pos_flat_fn
from konlpy.tag import Mecab
from data_utils.utils import Config


def load_data(data_path):
    """
    code from "https://github.com/changwookjun/Transformer/blob/25d9472155cb0788d11dbfe274526690915fe95e/data.py#L27"
    :param data_path:
    :return: list of train_input, train_label, eval_input, eval_label
    """
    # 판다스를 통해서 데이터를 불러온다.
    data_df = pd.read_csv(data_path, header=0)
    # 질문과 답변 열을 가져와 question과 answer에 넣는다.
    question, answer = list(data_df['Q']), list(data_df['A'])
    # skleran에서 지원하는 함수를 통해서 학습 셋과
    # 테스트 셋을 나눈다.
    tr_input, val_input, tr_label, val_label = train_test_split(question, answer, test_size=0.05,
                                                                        random_state=42)


    data_config = Config(json_path='./data_in/config.json')
    with open(data_config.train, mode='w', encoding='utf-8') as io:
        for input, label in zip(tr_input, tr_label):
            text = input+"\t"+label+"\n"
            io.write(text)

    with open(data_config.validation, mode='w', encoding='utf-8') as io:
        for input, label in zip(val_input, val_label):
            text = input+"\t"+label+"\n"
            io.write(text)

    total_courpus = question + answer
    # 그 값을 리턴한다.
    return total_courpus

def main():
    cwd = Path.cwd()
    print("cwd: ", cwd)
    full_path = cwd / 'data_in/Chatbot_data-master/ChatbotData.csv'
    print("full_path: ", full_path)
    total_courpus = load_data(data_path=full_path)
    print("total_courpus:", total_courpus)
    print("len(total_courpus): ", len(total_courpus))

    # extracting morph in sentences
    vocab = Vocabulary()
    token2idx = vocab.build_vocab(list_of_str=total_courpus, threshold=1, vocab_save_path="./data_in/token2idx_vocab.json", split_fn=mecab_token_pos_flat_fn) # Mecab().morphs
    print("token2idx: ", token2idx)



def gluonnlp_main():
    """
    추후 적용 예정
    :return:
    """
    import gluonnlp as nlp

    cwd = Path.cwd()
    full_path = cwd / 'data_in/Chatbot_data-master/ChatbotData.csv'
    tr_input, val_input, tr_label, val_label = load_data(data_path=full_path)

    total_input = tr_input + val_input
    mecab_tokenizer = Mecab()

    # extracting morph in sentences
    _list_of_tokens = [mecab_tokenizer.morphs(input_item) for input_item in total_input]
    list_of_tokens = []
    for _ in _list_of_tokens:
        list_of_tokens += _

    # making the vocab
    counter = nlp.data.count_tokens(itertools.chain.from_iterable(list_of_tokens))
    vocab = nlp.Vocab(counter=counter, min_freq=5, bos_token=None, eos_token=None)



if __name__ == '__main__':
    main()