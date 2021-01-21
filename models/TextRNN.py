# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np


class TextRNNConfig:
    sequence_length = 100
    vocab_size = 5000          # 词表大小
    embedding_dim = 300        # 词向量维度
    hidden_size = 128
    num_layers = 2
    dropout = 0.5
    num_classes = 10
    lr = 1e-3                  # 学习率
    batch_size = 32            # batch的大小    
    num_epochs = 10            # 数据训练轮数
    load_word2vec = False
    word2vec_path = ''
    require_improvement = 1000
    model_save_path = './ckpts/rnn_model.pth'


class TextRNN(nn.Module):
    '''BILSTM'''
    def __init__(self, config):
        super(TextRNN, self).__init__()
        if config.load_word2vec:
            embedding = np.load(config.word2vec_path)
            embedding = torch.tensor(embedding, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)


    def forward(self, x):
        embed = self.embedding(x)
        lstmout, _ = self.lstm(embed)    # https://blog.csdn.net/m0_45478865/article/details/104455978
        fc_input = lstmout[:, -1, :]     # 句子最后时刻的 hidden state
        out = self.fc(fc_input)
        return out
