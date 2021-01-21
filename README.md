# 文本分类
## 环境
python = 3.6  
pytorch = 1.6.0
## 数据集 
新闻分类数据集  
包含train文件50000条，val5000条，test10000条  
链接：https://pan.baidu.com/s/1Ej92nFjZwKRnhvCVY-YVXA   
提取码：8y27
## 算法和测试集表现
|算法|word|char|
|  ----  | ----  | ----  |
|textcnn|91.28%|90.86%|
|textrnn|91.27%|90.22%|
|textrcnn|93.16%|92.73%|
|rnn+attenton|92.41%|92.16%|
|dpcnn|92.27%|90.21%|

## 预测
demo.py是预测/部署阶段的代码

## 参考
- https://github.com/649453932/Chinese-Text-Classification-Pytorch
- https://github.com/gaussic/text-classification-cnn-rnn

