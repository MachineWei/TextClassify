# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class TextRCNNConfig:

    embedding_dim = 300        # 词向量维度
    hidden_size = 256
    num_layers = 1
    vocab_size = 5000          # 词表大小
    num_classes = 10
    sequence_length = 100      # 放入模型的句子长度
    batch_size = 32            # batch的大小
    num_epochs = 10            # 数据训练轮数
    lr = 1e-3                  # 学习率
    dropout_rate = 0.5         # dropout
    load_word2vec = False
    word2vec_path = ''
    require_improvement = 1000
    model_save_path = './ckpts/rcnn_model.pth'


class TextRCNN(nn.Module):

    def __init__(self, config):
        super(TextRCNN, self).__init__()
        if config.load_word2vec:
            embedding = np.load(config.word2vec_path)
            embedding = torch.tensor(embedding, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size, config.num_layers, bidirectional=True,
                            batch_first=True, dropout=config.dropout_rate)
        self.maxpool = nn.MaxPool1d(config.sequence_length)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embedding_dim, config.num_classes)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out),2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)  # 各维度调换shape(这里应该是变成了转置)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
