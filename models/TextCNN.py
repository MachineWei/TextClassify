# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class TextCNNConfig:

    sequence_length = 100      # 放入模型的句子长度
    vocab_size = 5000          # 词表大小

    embedding_dim = 300        # 词向量维度
    num_filters = 256          # 卷积核数目
    filter_size = [2, 3, 4]    # 卷积核尺寸
    num_classes = 10            # 类别数目
    
    batch_size = 32            # batch的大小
    num_epochs = 10             # 数据训练轮数
    lr = 1e-3                  # 学习率
    dropout_rate = 0.5         # dropout
    load_word2vec = False
    word2vec_path = ''
    require_improvement = 1000
    model_save_path = './ckpts/cnn_model.pth'


class TextCNN(nn.Module):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        if config.load_word2vec:
            embedding = np.load(config.word2vec_path)
            embedding = torch.tensor(embedding, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,  # 文本in_channels=1
                                              out_channels=config.num_filters,
                                              kernel_size=(k, config.embedding_dim)) for k in config.filter_size])

        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(config.num_filters * len(config.filter_size), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))                    # [32, 1, 100, 300]
        x = x.squeeze(3)
        x = F.max_pool1d(x, x.size(2))  # x.size(2) 对应参数：kernel_size，指窗口大小
        x = x.squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)  # 增加一个维度
        out_1, out_2, out_3 = [self.conv_and_pool(out, conv) for conv in self.convs]
        out = torch.cat([out_1, out_2, out_3], axis=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out