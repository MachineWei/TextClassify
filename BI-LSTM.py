# -*- coding: utf-8 -*-
"""
word2vec + BILSTM
"""

from data import TextData, create_embedding_matrix
from public import SimpleRNNConfig, plt_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, Bidirectional, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 读入训练数据
td = TextData()
(x_data, y_data, z_data), (x_labels, y_labels, z_labels) = td.load_idata()
word2id = td.word2id
cat2id = td.cat2id
num_classes=len(set(x_labels))
vocab_size = len(word2id)

# 将每条文本固定为相同长度
x_data = pad_sequences(x_data, SimpleRNNConfig.seq_length)   
x_labels = to_categorical(x_labels, num_classes=num_classes)

y_data = pad_sequences(y_data, SimpleRNNConfig.seq_length)   
y_labels = to_categorical(y_labels, num_classes=num_classes)

embedding_matrix = create_embedding_matrix('pretrain.model', word2id, SimpleRNNConfig.hidden_dims)  # pretrain.model词向量模型，根据需要自行下载或训练

# 构建模型
def w2v_bilstm(RNNConfig):
    model = Sequential()
    model.add(Embedding(vocab_size, RNNConfig.hidden_dims,
                        weights=[embedding_matrix],   # 嵌入词向量
                        input_length=RNNConfig.seq_length,
                        trainable=True))   
    
    model.add(Bidirectional(LSTM(RNNConfig.hidden_dims, return_sequences=True), merge_mode='concat'))
    
    model.add(Dense(RNNConfig.hidden_dims, activation='relu'))
    model.add(Dropout(RNNConfig.dropout))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    
    adam = Adam(lr=RNNConfig.learn_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    
    plot_model(model, to_file='./models/bilstm/w2v_bilstm.png', show_shapes=True, show_layer_names=False)
    model.summary()
    return model

model = w2v_bilstm(SimpleRNNConfig)
# 训练模型
history = model.fit(x_data, x_labels,
          batch_size=SimpleRNNConfig.batch_size, epochs=SimpleRNNConfig.epochs,
          validation_data=(y_data, y_labels),
          callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]
          )


# 模型评估
test_loss, test_acc = model.evaluate(y_data, y_labels)

# 模型预测
y_pred = model.predict_classes(y_data)

# 结果绘图至本地
pic_model(history,  y_labels, y_pred, plot_img_path='./models/bilstm')

## 模型保存
#save_path = './models/bilstm/w2v_bilstm.h5'
#model.save(save_path)
## 模型加载
#model = load_model(save_path)



