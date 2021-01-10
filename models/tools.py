# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


# 载入词向量模型
def create_embedding_matrix(word2vec_path, vocab_dict, embedding_dim):
    vocab_size = len(vocab_dict)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(word2vec_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, *vector = line.split()
            if word in vocab_dict:
                idx = vocab_dict[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix


# 让每条文本的长度相同，用0填充
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


# data_generator只是一种为了节约内存的数据方式
class data_generator:
    def __init__(self, data, seq_length, batch_size, tokenizer, shuffle=True):
        self.data = data
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:self.seq_length]
                x1, x2 = self.tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []


def plt_model(history, y_labels, y_pred, plot_img_path=None):
    '''
    可视化
    :param history:
    :param y_labels:
    :param y_pred:
    :param plot_img_path:
    :return:
    '''
    # 混淆矩阵
    y_labels_ = np.argmax(y_labels, axis=1)
    con_mat = confusion_matrix(y_labels_, y_pred)
    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)
    sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
    plt.ylim(0, len(set(y_labels_)))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    if plot_img_path:
        plt.savefig(os.path.join(plot_img_path, 'confusion_matrix.png'), dpi=200)
    plt.show()

    # 损失值曲线
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    if plot_img_path:
        plt.savefig(os.path.join(plot_img_path, 'loss.png'), dpi=200)
    plt.show()

