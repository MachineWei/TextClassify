# -*- coding: utf-8 -*-
"""
Attention+CNN
"""

from .Attention import Self_Attention
from keras import Input
from keras.models import Model
from keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, Dense, Dropout
from keras.optimizers import Adam


class AttentionCNNConfig:
    seq_length = 500
    batch_size = 128
    embedding_dims = 64
    filters = 128
    kernel_size = 8
    hidden_dims = 64
    epochs = 8
    dropout = 0.5
    learn_rate = 1e-3


class AttentionCNN(AttentionCNNConfig):

    def __init__(self, vocab_size, num_classes):
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def build_model(self, embedding_matrix=None):
        # 构建模型
        x_input = Input(shape=(self.seq_length,), dtype='int32')
        embeddings = Embedding(input_dim=self.vocab_size,
                               output_dim=self.embedding_dims,
                               input_length=self.seq_length)(x_input)
        seq = Self_Attention(self.embedding_dims)(embeddings)
        seq = Conv1D(self.filters, self.kernel_size, padding='valid',
                     activation='relu', strides=1)(seq)
        seq = GlobalAveragePooling1D()(seq)
        seq = Dropout(self.dropout)(seq)
        outputs = Dense(self.num_classes, activation='softmax')(seq)

        model = Model(inputs=x_input, outputs=outputs)

        adam = Adam(lr=self.learn_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam, metrics=['accuracy'])

        model.summary()
        # plot_model(model, to_file='./models/attention_cnn/attention_cnn.png', show_shapes=True, show_layer_names=False)
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