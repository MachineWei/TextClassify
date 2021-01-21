# -*- coding: utf-8 -*-

import numpy as np

def batch_iter(x, y, batch_size=64):
    """
    生成批次数据
    """
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.RandomState(seed = 42).permutation(np.arange(data_len))  # 可复现
    # indices = np.random.permutation(np.arange(data_len))
    x_shuffle = np.array(x)[indices]
    y_shuffle = np.array(y)[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
