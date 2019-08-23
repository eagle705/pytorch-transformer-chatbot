# PyTorch_Transformer_Chatbot

Simple Korean Generative Chatbot Implementation based on new PyTorch Transformer API (PyTorch v1.2 / Python 3.x)

![transformer_fig](./assets/transformer_fig.png)

### ToDo
- Beam Search
- Search hyperparams
- Attention Visualization
- Char-level transformer

### Transformer API core logic
- Padding masking이 매우 편함
- decoder input의 future token을 못보게 하기 위한 masking은 함수로 제공함
- Transformer의 input, output dim 순서는 [Sequnece, Batch, Embedding Dimension]으로 되어있어서 Transpose 해줘야함
- 아쉽지만 Transformer API에서 Attention weight dict을 제공해주진 않음

```python
def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor) -> torch.Tensor:
    x_enc_embed = self.input_embedding(enc_input.long())
    x_dec_embed = self.input_embedding(dec_input.long())

    # Masking
    src_key_padding_mask = enc_input == self.vocab.PAD_ID # tensor([[False, False, False,  True,  ...,  True]])
    tgt_key_padding_mask = dec_input == self.vocab.PAD_ID
    memory_key_padding_mask = src_key_padding_mask
    tgt_mask = self.transfomrer.generate_square_subsequent_mask(dec_input.size(1))

    # einsum ref: https://pytorch.org/docs/stable/torch.html#torch.einsum
    # https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
    x_enc_embed = torch.einsum('ijk->jik', x_enc_embed)
    x_dec_embed = torch.einsum('ijk->jik', x_dec_embed)


    # transformer ref: https://pytorch.org/docs/stable/nn.html#torch.nn.Transformer
    feature = self.transfomrer(src = x_enc_embed,
                               tgt = x_dec_embed,
                               src_key_padding_mask = src_key_padding_mask,
                               tgt_key_padding_mask = tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               tgt_mask = tgt_mask.to(device)) # src: (S,N,E) tgt: (T,N,E)

    logits = self.proj_vocab_layer(feature)
    logits = torch.einsum('ijk->jik', logits)

    return logits
```

### Experiments

- train set에 대해서는 Overfit으로 95%의 정확도를 보이지만, val set에 대해서는 낮음 (예시로 공개하기 애매할정도)
- padding 값은 acc, loss 계산에서 모두 제외함

```
input:  [['나/NP', '를/JKO', '사랑/NNG', '한/XSA+ETM', '그/MM', '사람/NNG', '에게/JKB', '해/VV+EC', '줄/VX+ETM', '수/NNB', '있/VV', '는/ETM', '것/NNB', '<pad>', '<pad>'], ['맥주/NNG', '한/MM', '잔/NNG', '해야지/VV+EC', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']]
pred:  [['천천히/MAG', '잊어/NNP', '가/JKS', '요/JX', './SF', '</s>', '들/VV', '노력/NNG', '요/JX', '은/JX', '도/JX', '이/JKS', '도/JX', '이/JKS', '이/JKS'], ['적당히/MAG', '드세요/VV+EP+EF', './SF', '</s>', '에/JKB', '에/JKB', '드세요/VV+EP+EF', '드세요/VV+EP+EF', '구경/NNG', '드세요/VV+EP+EF', '에/JKB', '드세요/VV+EP+EF', '가리/VV', '드세요/VV+EP+EF', '드세요/VV+EP+EF']]
real:  [['천천히/MAG', '잊어/NNP', '가/JKS', '요/JX', './SF', '</s>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'], ['적당히/MAG', '드세요/VV+EP+EF', './SF', '</s>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']]
```

- Inference test 결과
    - train set에 있는건 잘 대답함
    - 얼추 잘(?) 입력하면 잘 대답함
    - "unk" 뜨거나 토큰 하나 바뀌어도 대답이 바뀜 ㅠ

```
문장을 입력하세요: 배고파
input:  [['배고파/VA+EC', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']]
pred:  [['얼른/MAG', '맛난/VA+ETM', '음식/NNG', '드세요/VV+EP+EF', './SF', '</s>']]
pred_str:  얼른 맛난 음식 드세요.

문장을 입력하세요: 너 누구야
input:  [['너/NP', '누구/NP', '야/VCP+EF', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']]
pred:  [['저/NP', '는/JX', '마음/NNG', '을/JKO', '이어주/VV', '는/ETM', '위/NNG', '로/JKB', '봇/NNG', '입니다/VCP+EF', './SF', '</s>']]
pred_str:  저는 마음을 이어주는 위로봇입니다.

문장을 입력하세요: 안녕
input:  [['안녕/IC', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']]
pred:  [['안녕/NNG', '하/XSV', '세요/EP+EF', './SF', '</s>']]
pred_str:  안녕하세요.

문장을 입력하세요: 졸리다 이제 자야지
input:  [['<unk>', '다/EF', '이제/MAG', '자/VV', '야지/EC', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']]
pred:  [['너무/MAG', '걱정/NNG', '하/XSV', '지/EC', '마세요/VX+EP+EF', './SF', '</s>']]
pred_str:  너무 걱정하지 마세요.
```

### 실행순서
- 테스트만 할경우 ```inference.py```만 실행

```bash
python build_vocab.py # 사전 구축
python train.py # 학습
python inference.py # 테스트
```

### Requirements

```bash
pip install mxnet
pip install gluonnlp
pip install konlpy
pip install python-mecab-ko
pip install chatspace
pip install tb-nightly
pip install future
pip install pathlib
```


### Reference Repositories
- [Chatbot Dataset by songys](https://github.com/songys/Chatbot_data)
- [Warmup Scheduler by ildoonet](https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py)
- [NLP implementation by aisolab](https://github.com/aisolab/nlp_implementation)
- [Transformer implementation by changwookjun](https://github.com/changwookjun/Transformer)
- [PyTorch_BERT_Implementation by codertimo](https://github.com/codertimo/BERT-pytorch)
- [pingpong-ai/chatspace](https://github.com/pingpong-ai/chatspace/tree/master)
- [PyTorch Seq2Seq Tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/seq2seq_translation_tutorial.ipynb#scrollTo=OXkt42mheogQ)
