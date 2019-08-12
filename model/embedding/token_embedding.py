from __future__ import absolute_import, division, print_function, unicode_literals

from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, pad_id):
        super(TokenEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)

    def forward(self, x):
        x_embed = self.token_embedding(x)
        return x_embed

def main():
    print("TokenEmbedding")

if __name__ == '__main__':
    main()