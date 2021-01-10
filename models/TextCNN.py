# -*- coding: utf-8 -*-
'''
TextCNN新闻文本分类
'''

from keras import Input
# from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import Embedding, Conv1D, MaxPool1D, concatenate, Flatten
from keras.optimizers import Adam


class TCNNConfig:
    seq_length = 500
    batch_size = 128
    embedding_dims = 64
    filters = 42
    kernel_sizes = [2, 3, 4]
    hidden_dims = 64
    epochs = 5
    dropout = 0.5
    learn_rate = 1e-3


class TextCNN(TCNNConfig):

    def __init__(self, vocab_size, num_classes):
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def build_model(self, embedding_matrix=None):
        '''构建模型'''
        x_input = Input(shape=(self.seq_length,))
        if embedding_matrix:
            x_emb = Embedding(input_dim=self.vocab_size,
                              output_dim=self.embedding_dims,
                              weights=[embedding_matrix],  # 嵌入词向量
                              input_length=self.seq_length)(x_input)
        else:
            x_emb = Embedding(input_dim=self.vocab_size,
                              output_dim=self.embedding_dims,
                              input_length=self.seq_length)(x_input)

        pool_output = []
        for kernel_size in self.kernel_sizes:
            c = Conv1D(filters=self.filters,
                       kernel_size=kernel_size,
                       padding='valid',
                       activation='relu',
                       strides=1)(x_emb)
            p = MaxPool1D(pool_size=int(c.shape[1]))(c)
            pool_output.append(p)
        pool_output = concatenate([p for p in pool_output])

        x_flatten = Flatten()(pool_output)
        y = Dense(self.num_classes, activation='softmax')(x_flatten)

        model = Model([x_input], outputs=[y])

        adam = Adam(lr=self.learn_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam, metrics=['accuracy'])

        model.summary()
        # plot_model(model, to_file='../img/text_cnn/text_cnn.png', show_shapes=True, show_layer_names=False)  # 很可能会出错
        self.model = model

    def train(self, x_data, x_labels, y_data, y_labels):
        history = self.model.fit(x_data, x_labels,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=(y_data, y_labels),
                                 verbose=1,
                                 shuffle=True)
        return history

    def evaluate(self, z_data, z_labels):
        return self.model.evaluate(z_data, z_labels)

    def predict(self, y_data):
        return self.model.predict(y_data)