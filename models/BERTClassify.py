# -*- coding: utf-8 -*-
"""
Bert文本分类
"""

from keras import Input
from keras.models import Model
from keras.layers import Lambda, Dropout, Dense
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, load_vocabulary
from keras.optimizers import Adam


class OurTokenizer(Tokenizer):
    # 重写tokenizer
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


class BertConfig:
    # 超参数
    seq_length = 500
    batch_size = 32
    droupout = 0.5
    learn_rate = 1e-5
    epochs = 5

    # 预训练模型目录
    config_path = r"./ckpt/pre_bert_model/chinese_L-12_H-768_A-12/bert_config.json"
    checkpoint_path = r"./ckpt/pre_bert_model/chinese_L-12_H-768_A-12/bert_model.ckpt"
    dict_path = r"./ckpt/pre_bert_model/chinese_L-12_H-768_A-12/vocab.txt"


class BERTClassify(BertConfig):

    def __init__(self, num_classes):
        self.num_classes = num_classes
        token_dict = load_vocabulary(self.dict_path)
        self.tokenizer = OurTokenizer(token_dict)

    def build_model(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path,
                                                        self.checkpoint_path,
                                                        False, trainable=True)

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)   # 取出[CLS]对应的向量用来做分类
        x = Dropout(BertConfig.droupout)(x)
        p = Dense(self.num_classes, activation='softmax')(x)

        model = Model([x1_in, x2_in], p)
        model.compile(
                        loss='categorical_crossentropy',
                        optimizer=Adam(BertConfig.learn_rate),
                        metrics=['accuracy']
                     )
        model.summary()
        self.model = model

    def train(self, train_D, valid_D):
        history = self.model.fit_generator(
                                        train_D.__iter__(),
                                        steps_per_epoch=len(train_D),
                                        epochs=BertConfig.epochs,
                                        validation_data=valid_D.__iter__(),
                                        validation_steps=len(valid_D)
                                      )
        return history

    def predict(self, test_D):
        return self.model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)


