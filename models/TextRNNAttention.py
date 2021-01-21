# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class TextRNNAttentionConfig:

    sequence_length = 100
    vocab_size = 5000          # 词表大小
    embedding_dim = 300        # 词向量维度
    hidden_size = 128
    hidden_size2 = 64
    num_layers = 2
    dropout = 0.5
    num_classes = 10
    batch_size = 32            # batch的大小       
    lr = 1e-3                 # 学习率
    num_epochs = 10           # 数据训练轮数
    load_word2vec = False
    word2vec_path = ''
    require_improvement = 1000
    model_save_path = './ckpts/rt_model.pth'


class TextRNNAttention(nn.Module):
    def __init__(self, config):
        super(TextRNNAttention, self).__init__()
        if config.load_word2vec:
            embedding = np.load(config.word2vec_path)
            embedding = torch.tensor(embedding, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        # self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc2 = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstmout, _ = self.lstm(emb)

        # attention # https://www.cnblogs.com/DjangoBlog/p/9504771.html
        M = self.tanh1(lstmout)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = lstmout * alpha
        out = torch.sum(out, axis=1)

        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out