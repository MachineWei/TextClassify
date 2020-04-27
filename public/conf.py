# -*- coding: utf-8 -*-


class SimpleRNNConfig:
    seq_length = 500
    batch_size = 128
    hidden_dims = 64
    epochs = 5
    dropout = 0.5
    learn_rate = 1e-3
    

class SimpleCNNConfig:
    seq_length = 500
    batch_size = 128
    embedding_dims = 64
    filters = 128
    kernel_size = 8
    hidden_dims = 64
    epochs = 5
    dropout = 0.5
    learn_rate = 1e-3


class TCNNConfig:
    seq_length = 500
    batch_size = 128
    embedding_dims = 64
    filters = 42
    kernel_sizes = [2, 3, 4]
    hidden_dims = 64
    epochs = 5
    dropout = 0.5
    learn_rate = 1e-3


class RCNNConfig:
    seq_length = 500
    batch_size = 128
    hidden_dims = 64
    epochs = 5
    dropout = 0.5
    learn_rate = 1e-3


class AttentionCNNConfig:
    seq_length = 500
    batch_size = 128
    embedding_dims = 64
    filters = 128
    kernel_size = 8
    hidden_dims = 64
    epochs = 8
    dropout = 0.5
    learn_rate = 1e-3




