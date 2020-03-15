# -*- coding: utf-8 -*-

import os

cur_path = os.path.abspath(os.path.dirname(__file__))
DataDir = os.path.join(cur_path, 'data')
ModelDir = os.path.join(cur_path, 'models')

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
    stop_words = os.path.join(segment_dir, 'stop_words.txt')


class CNNConfig(BasePath):
    
    seq_length = 500
    batch_size = 128
    embedding_dims = 64
    filters = 128
    kernel_size = 8
    hidden_dims = 64
    
    epochs = 5
    dropout = 0.5
    learn_rate = 1e-3
    
    save_path = os.path.join(ModelDir, 'simple_cnn')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_path = os.path.join(save_path, 'simple_cnn.png')
    model_path = os.path.join(save_path, 'simple_cnn.h5')


class TCNNConfig(BasePath):
    
    seq_length = 500
    batch_size = 128
    embedding_dims = 64
    filters = 42
    kernel_sizes = [2,3,4]
    hidden_dims = 64
    
    epochs = 5
    dropout = 0.5
    learn_rate = 1e-3
    
    save_path = os.path.join(ModelDir, 'text_cnn')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_path = os.path.join(save_path, 'text_cnn.png')
    model_path = os.path.join(save_path, 'text_cnn.h5')


class RNNConfig(BasePath):
    
    rnn = 'lstm'
    
    seq_length = 500
    batch_size = 128
    hidden_dims = 64

    epochs = 5
    dropout = 0.5
    learn_rate = 1e-3

    save_path = os.path.join(ModelDir, 'simple_rnn')
    if not os.path.exists(save_path):
        os.mkdir(save_path)    
    img_path = os.path.join(save_path, 'model.png')
    model_path = os.path.join(save_path, 'simple_rnn.h5')







