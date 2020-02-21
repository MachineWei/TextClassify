# -*- coding: utf-8 -*-
'''
cnn新闻文本分类
'''

from data_utils import TextData
from conf import TCNNConfig
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D
from keras.optimizers import Adam

# 读入训练数据
td = TextData()
(x_data, y_data, z_data), (x_labels, y_labels, z_labels) = td.load_idata()
word2id = td.word2id
cat2id = td.cat2id
num_classes=len(set(x_labels))
vocab_size = len(word2id)

# 将每条文本固定为相同长度
x_data = pad_sequences(x_data, TCNNConfig.seq_length)   
x_labels = to_categorical(x_labels, num_classes=num_classes)

y_data = pad_sequences(y_data, TCNNConfig.seq_length)   
y_labels = to_categorical(y_labels, num_classes=num_classes)

# 构建模型
model = Sequential()
model.add(Embedding(vocab_size, TCNNConfig.embedding_dims,
                    input_length=TCNNConfig.seq_length))

model.add(Conv1D(TCNNConfig.filters, TCNNConfig.kernel_size, padding='valid',
                 activation='relu', strides=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(TCNNConfig.hidden_dims))
model.add(Dropout(TCNNConfig.dropout))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

adam = Adam(lr=TCNNConfig.learn_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=adam, metrics=['accuracy'])

model.summary()

# 训练模型
history = model.fit(x_data, x_labels,
          batch_size=TCNNConfig.batch_size, epochs=TCNNConfig.epochs,
          validation_data=(y_data, y_labels))

# 模型保存
model.save(TCNNConfig.save_path)

# 模型加载
model = load_model(TCNNConfig.save_path)

# 模型评估
test_loss, test_acc = model.evaluate(y_data, y_labels)

# 模型预测
y_pred = model.predict_classes(y_data)


# 结果可视化
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# 混淆矩阵
y_labels_ = np.argmax(y_labels, axis=1)
con_mat = confusion_matrix(y_labels_, y_pred)
con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis] # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=2)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_norm, annot=True, cmap='Blues')

plt.ylim(0, len(set(y_labels_)))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# 损失值曲线
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.show()






