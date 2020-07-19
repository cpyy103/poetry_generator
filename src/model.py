#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from src.config import Config


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)    # 词向量层
        self.num_layers = Config['num_layers']
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        sequence_length, batch_size = x.size()
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # embeds shape (sequence_length, batch_size, embedding_dim )
        embeds = self.embedding(x)

        # output shape (sequence_length, batch_size, hidden_dim*(2 if bidirectional else 1) )
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # output shape (sequence_length*batch, vocab_size)
        output = self.linear(output.view(sequence_length*batch_size, -1))

        return output, hidden


if __name__ == '__main__':
    model = PoetryModel(1000, 100, 100)
    print(model)











