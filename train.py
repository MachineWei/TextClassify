# -*- coding: utf-8 -*-
'''
新闻文本分类(TextCNN, BiLSTM, AttentionCNN)
'''

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from data.data_utils import LoadData
from models import TextCNN, BiLSTM, AttentionCNN
from models.tools import plt_model, create_embedding_matrix

# 自定义模型
MODEL = TextCNN

# 读入训练数据
td = LoadData()
(x_data, y_data, z_data), (x_labels, y_labels, z_labels) = td.load_idata()
word2id = td.word2id
cat2id = td.cat2id
num_classes = len(set(x_labels))
vocab_size = len(word2id)
# embedding_matrix = create_embedding_matrix('pretrain.model', word2id, MODEL.hidden_dims)  # pretrain.model词向量模型，根据需要自行下载或训练

# 将每条文本固定为相同长度
x_data = pad_sequences(x_data, MODEL.seq_length)
x_labels = to_categorical(x_labels, num_classes=num_classes)
y_data = pad_sequences(y_data, MODEL.seq_length)
y_labels = to_categorical(y_labels, num_classes=num_classes)

# 定义模型
model = MODEL(vocab_size, num_classes)
model.build_model()

# 训练模型
history = model.train(x_data, x_labels, y_data, y_labels)

# 模型评估
test_loss, test_acc = model.evaluate(z_data, z_labels)

# 模型预测
y_pred = model.predict(y_data)
y_pred = np.argmax(y_pred, axis=1)

# 结果绘图至本地
plt_model(history,  y_labels, y_pred, plot_img_path='./models/text_cnn')


## 模型保存
#save_path = './models/text_cnn/text_cnn.h5'
#model.save(save_path)
## 模型加载
#model = load_model(save_path)







