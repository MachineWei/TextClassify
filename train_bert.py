# -*- coding: utf-8 -*-

import numpy as np
from models.tools import data_generator
from data.data_utils import LoadData
from keras.utils import to_categorical
from models import BERTClassify

# 载入数据
td = LoadData()
(x_data, y_data, z_data), (x_labels, y_labels, z_labels) = td.load_data()  # 注意load的不是id
word2id = td.word2id
cat2id = td.cat2id
num_classes=len(set(x_labels))
vocab_size = len(word2id)

# bert输入数据转换
x_labels = [cat2id[i] for i in x_labels]
y_labels = [cat2id[i] for i in y_labels]
z_labels = [cat2id[i] for i in z_labels]
x_labels = to_categorical(x_labels, num_classes=num_classes)
y_labels = to_categorical(y_labels, num_classes=num_classes)
z_labels = to_categorical(z_labels, num_classes=num_classes)

# 数据格式转换成模型需要的输入格式
train_data = [np.array([x_data[i].replace(' ', ''), x_labels[i]]) for i in range(len(x_labels))]
valid_data = [np.array([y_data[i].replace(' ', ''), y_labels[i]]) for i in range(len(y_labels))]
test_data = [np.array([z_data[i].replace(' ', ''), z_labels[i]]) for i in range(len(z_labels))]

# 开始训练
bert_model = BERTClassify(num_classes)
train_D = data_generator(train_data, bert_model.seq_length, bert_model.batch_size, bert_model.tokenizer)
valid_D = data_generator(valid_data, bert_model.seq_length, bert_model.batch_size, bert_model.tokenizer)
test_D = data_generator(test_data,  bert_model.seq_length, bert_model.batch_size, bert_model.tokenizer)
bert_model.build_model()
# xunlian
history = bert_model.train(train_D, valid_D)

# 模型预测
test_model_pred = bert_model.predict(test_D)
test_pred = [np.argmax(x) for x in test_model_pred]

