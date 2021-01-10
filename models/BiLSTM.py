# -*- coding: utf-8 -*-
"""
word2vec + BILSTM
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding, LSTM, Bidirectional, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


class RNNConfig:
    seq_length = 500
    batch_size = 128
    hidden_dims = 64
    epochs = 5
    dropout = 0.5
    learn_rate = 1e-3


class BiLSTM(RNNConfig):

    def __init__(self, vocab_size, num_classes):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.model = self.build_model()

    # 构建模型
    def build_model(self, embedding_matrix=None):
        model = Sequential()
        if embedding_matrix:
            model.add(Embedding(self.vocab_size, self.hidden_dims,
                                weights=[embedding_matrix],   # 嵌入词向量
                                input_length=self.seq_length,
                                trainable=True))
        else:
            model.add(Embedding(self.vocab_size, self.hidden_dims,
                                input_length=self.seq_length))

        model.add(Bidirectional(LSTM(self.hidden_dims, return_sequences=True), merge_mode='concat'))

        model.add(Dense(self.hidden_dims, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax'))

        adam = Adam(lr=self.learn_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam, metrics=['accuracy'])

        # plot_model(model, to_file='./models/bilstm/w2v_bilstm.png', show_shapes=True, show_layer_names=False)
        model.summary()
        self.model = model

    def train(self, x_data, x_labels, y_data, y_labels):
        history = self.model.fit(x_data, x_labels,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=(y_data, y_labels),
                                 verbose=1,
                                 shuffle=True,
                                 callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])
        return history

    def evaluate(self, z_data, z_labels):
        return self.model.evaluate(z_data, z_labels)

    def predict(self, y_data):
        return self.model.predict(y_data)


