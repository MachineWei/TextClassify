# 文本分类
## 数据集 
新闻分类数据集  
包含{'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}十个类别，其中train文件50000条，val5000条，test10000条。  

## 算法实现
- bayes
- rnn
- cnn
- w2v+lstm

## 代码说明
```
TextClassify/  
|-- data/                    # 数据文件
|   |-- raw_data             # 原始数据
|   |   |-- cnews.test.txt
|   |   |-- cnews.train.txt
|   |   |-- cnews.val.txt
|   |   |-- readme.md
|   |-- segment_data         # 经过处理后的数据（分词）
|   |   |-- test.txt
|   |   |-- train.txt
|   |   |-- val.txt
|   |   |-- vocab.txt
|   |   |-- stopwords.txt
|-- model/                   # 存储训练结果文件（持久化模型、结构图、混淆矩阵、损失曲线）
|   |-- simple_cnn/  
|   |   |-- confusion_matrix.png
|   |   |-- loss.png
|   |   |-- simple_cnn.png  
|   |   |-- simple_cnn.h5
|-- pretrain.model           # 词向量模型（根据需要去别处下载）
|-- conf.py                  # 模型配置文件
|-- data_utils.py            # 数据预处理模块
|-- pic.py                   # 作图
|-- SimpleCNN.py             # 基础版cnn
|-- SimpleRNN.py             # 基础版rnn
|-- SklearnBayes.py          # 贝叶斯分类
|-- TextCNN.py               # textcnn
|-- Word2VecLSTM.py          # word2vec + lstm
```


