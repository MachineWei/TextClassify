# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class TextDPCNNConfig:

    sequence_length = 100      # 放入模型的句子长度
    vocab_size = 5000          # 词表大小
    embedding_dim = 300        # 词向量维度
    num_filters = 250
    num_layers = 2
    dropout = 0.5
    num_classes = 10
    lr = 1e-3                  # 学习率
    batch_size = 32            # batch的大小    
    num_epochs = 10            # 数据训练轮数
    load_word2vec = False
    word2vec_path = ''
    require_improvement = 1000
    model_save_path = './ckpts/dpcnn_model.pth'


class DPCNN(nn.Module):

    def __init__(self, config):
        super(DPCNN, self).__init__()
        if config.load_word2vec:
            embedding = np.load(config.word2vec_path)
            embedding = torch.tensor(embedding, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embedding_dim), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.relu = nn.ReLU()
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.conv_region(x)

        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)

        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        x = x + px
        return x

