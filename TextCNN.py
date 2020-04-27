# -*- coding: utf-8 -*-
'''
TextCNN新闻文本分类
'''

import numpy as np
from data import TextData
from public import TCNNConfig, plt_model
from keras import Input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout
from keras.layers import Embedding, Conv1D, MaxPool1D, concatenate, Flatten
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


def text_cnn(CNNConfig):
    # 构建模型
    x_input = Input(shape=(TCNNConfig.seq_length,))
    x_emb = Embedding(input_dim=vocab_size, 
                      output_dim=TCNNConfig.embedding_dims, 
                      input_length=TCNNConfig.seq_length)(x_input)
    pool_output = []
    for kernel_size in TCNNConfig.kernel_sizes:
        c = Conv1D(filters=TCNNConfig.filters, 
                   kernel_size=kernel_size, 
                   padding='valid',
                   activation='relu',                    
                   strides=1)(x_emb)
        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)
    pool_output = concatenate([p for p in pool_output])

    x_flatten = Flatten()(pool_output)  
    y = Dense(num_classes, activation='softmax')(x_flatten)  

    model = Model([x_input], outputs=[y])
    
    adam = Adam(lr=TCNNConfig.learn_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    
    model.summary()
    
    plot_model(model, to_file='./models/text_cnn/text_cnn.png', show_shapes=True, show_layer_names=False)
    
    model.summary()
    
    return model
    

model = text_cnn(TCNNConfig)

# 训练模型
history = model.fit(x_data, x_labels,
          batch_size=TCNNConfig.batch_size, epochs=TCNNConfig.epochs,
          validation_data=(y_data, y_labels))

# 模型评估
test_loss, test_acc = model.evaluate(y_data, y_labels)

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


