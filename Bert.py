# -*- coding: utf-8 -*-
"""
Bert文本分类
（高性能服务器请架起）
"""

import numpy as np
from data import TextData, data_generator
from public import BertConfig
from keras import Input
from keras.models import Model
from keras.layers import Lambda, Dropout, Dense
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, load_vocabulary
from keras.optimizers import Adam
from keras.utils import to_categorical

# 读入训练数据
td = TextData()
# 这里x_data, y_data, z_data应为原始数据，未做分词处理
(x_data, y_data, z_data), (x_labels, y_labels, z_labels) = td.load_data()
word2id = td.word2id
cat2id = td.cat2id
num_classes=len(set(x_labels))
vocab_size = len(word2id)

x_labels = [cat2id[i] for i in x_labels]
y_labels = [cat2id[i] for i in y_labels]
x_labels = to_categorical(x_labels, num_classes=num_classes)
y_labels = to_categorical(y_labels, num_classes=num_classes)

# 数据格式转换成模型需要的输入格式
train_data = [np.array([x_data[i], x_labels[i]]) for i in range(len(x_labels))]
valid_data = [np.array([y_data[i], y_labels[i]]) for i in range(len(y_labels))]

# 读取字典
token_dict = load_vocabulary(BertConfig.dict_path)

#重写tokenizer        
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示
        return R
tokenizer = OurTokenizer(token_dict)


def bert_model(BertConfig):
    bert_model = load_trained_model_from_checkpoint(BertConfig.config_path, 
                                                    BertConfig.checkpoint_path, 
                                                    False, trainable=True)
    
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    
    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
    x = Dropout(BertConfig.droupout)(x)
    p = Dense(num_classes, activation='softmax')(x)
    
    model = Model([x1_in, x2_in], p)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(BertConfig.learn_rate),
        metrics=['accuracy']
    )
    model.summary()
    return model


model = bert_model(BertConfig)

train_D = data_generator(train_data, BertConfig.seq_length, BertConfig.batch_size, tokenizer)
valid_D = data_generator(valid_data, BertConfig.seq_length, BertConfig.batch_size, tokenizer)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=BertConfig.epochs,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)







