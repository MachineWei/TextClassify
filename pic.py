# -*- coding: utf-8 -*-
'''
# 结果可视化
'''

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def pic_model(history, y_labels, y_pred, plot_img_path=None):
    # 混淆矩阵
    y_labels_ = np.argmax(y_labels, axis=1)
    con_mat = confusion_matrix(y_labels_, y_pred)
    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis] # 归一化
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
    plt.legend(["train","test"],loc="upper left")
    if plot_img_path:
        plt.savefig(os.path.join(plot_img_path, 'loss.png'), dpi=200)    
    plt.show()







