# -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer


class Self_Attention(Layer):
  
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)
  
    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
  
        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它
  
    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
  
        print("WQ.shape", WQ.shape)  
        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64**0.5)   # 64为自定，原是一个与Q和K相关的尺度标度
        QK = K.softmax(QK)
        print("QK.shape",QK.shape)
  
        V = K.batch_dot(QK,WV)
        return V
  
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)    
   
    
    