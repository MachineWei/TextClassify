# -*- coding: utf-8 -*-
"""
rnn新闻文本分类
"""

from data import TextData
from public import SimpleRNNConfig, plt_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Embedding, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 读入训练数据
td = TextData()
(x_data, y_data, z_data), (x_labels, y_labels, z_labels) = td.load_idata()
word2id = td.word2id
cat2id = td.cat2id
num_classes = len(set(x_labels))
vocab_size = len(word2id)

# 将每条文本固定为相同长度
x_data = pad_sequences(x_data, SimpleRNNConfig.seq_length)
x_labels = to_categorical(x_labels, num_classes=num_classes)

y_data = pad_sequences(y_data, SimpleRNNConfig.seq_length)
y_labels = to_categorical(y_labels, num_classes=num_classes)


# 构建模型
def simple_rnn(SimpleRNNConfig):

    model = Sequential()
    model.add(Embedding(vocab_size, SimpleRNNConfig.hidden_dims,
                        input_length=SimpleRNNConfig.seq_length))
    model.add(LSTM(SimpleRNNConfig.hidden_dims))
    
    model.add(Dense(SimpleRNNConfig.hidden_dims, activation='relu'))
    model.add(Dropout(SimpleRNNConfig.dropout))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    adam = Adam(lr=SimpleRNNConfig.learn_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    
    plot_model(model, to_file='./models/simple_rnn/simple_rnn.png', show_shapes=True, show_layer_names=False)
    model.summary()
    
    return model


model = simple_rnn(SimpleRNNConfig)

# 训练模型
history = model.fit(x_data, x_labels,
          batch_size=SimpleRNNConfig.batch_size, epochs=SimpleRNNConfig.epochs,
          validation_data=(y_data, y_labels),
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)]   # 当val-loss不再提升时停止训练
          )

# 模型评估
test_loss, test_acc = model.evaluate(z_data, z_labels)

# 模型预测
y_pred = model.predict_classes(y_data)

# 结果绘图至本地
plt_model(history,  y_labels, y_pred, plot_img_path='./models/simple_rnn')

## 模型保存
#save_path = './models/simple_rnn/simple_rnn.h5'
#model.save(save_path)
## 模型加载
#model = load_model(save_path)

