# -*- coding: utf-8 -*-

import os

cur_path = os.path.abspath(os.path.dirname('__file__'))
DataDir = os.path.join(cur_path, 'data')
ModelDirt = os.path.join(cur_path, 'models')

class BasePath:
    base_dir = os.path.join(DataDir,'raw_data')
    raw_train = os.path.join(base_dir, 'cnews.train.txt')
    raw_test = os.path.join(base_dir, 'cnews.test.txt')
    raw_val = os.path.join(base_dir, 'cnews.val.txt')
    
    segment_dir = os.path.join(DataDir,'segment_data')
    train_dir = os.path.join(segment_dir, 'train.txt')
    test_dir = os.path.join(segment_dir, 'test.txt')
    val_dir = os.path.join(segment_dir, 'val.txt')
    vocab_dir = os.path.join(segment_dir, 'vocab.txt')
    
    models_dir = os.path.join(DataDir, 'models')


class TCNNConfig(BasePath):
    model_name = 'keras_cnn.h5'
    
    seq_length = 500
    vocab_size = 5000
    batch_size = 128
    embedding_dims = 64
    filters = 128
    kernel_size = 8
    hidden_dims = 64
    
    epochs = 5
    dropout = 0.5
    learn_rate = 1e-3
    
    save_path = os.path.join(ModelDirt, model_name)


class TRNNConfig(BasePath):
    model_name = 'keras_rnn.h5'

    seq_length = 500
    vocab_size = 5000
    batch_size = 128
    hidden_dims = 64

    epochs = 5
    dropout = 0.5
    learn_rate = 1e-3

    save_path = os.path.join(ModelDirt, model_name)







