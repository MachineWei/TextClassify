# -*- coding: utf-8 -*-

import os

BaseDir = r'E:\MySVN\Github\TextClassify\data'
ModelDirt = r'E:\MySVN\Github\TextClassify\models'

class BasePath:
    base_dir = os.path.join(BaseDir,'raw_data')
    raw_train = os.path.join(base_dir, 'cnews.train.txt')
    raw_test = os.path.join(base_dir, 'cnews.test.txt')
    raw_val = os.path.join(base_dir, 'cnews.val.txt')
    
    segment_dir = os.path.join(BaseDir,'segment_data')
    train_dir = os.path.join(segment_dir, 'train.txt')
    test_dir = os.path.join(segment_dir, 'test.txt')
    val_dir = os.path.join(segment_dir, 'val.txt')
    vocab_dir = os.path.join(segment_dir, 'vocab.txt')
    
    models_dir = os.path.join(BaseDir, 'models')


class TCNNConfig(BasePath):
    name = 'keras_cnn.h5'
    max_len = 500
    batch_size = 128
    embedding_dims = 256
    filters = 128
    kernel_size = 8
    hidden_dims = 64
    epochs = 5
    dropout = 0.5
    save_path = os.path.join(ModelDirt, name)

















