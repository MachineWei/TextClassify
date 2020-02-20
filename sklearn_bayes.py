# -*- coding: utf-8 -*-
"""
贝叶斯新闻文本分类
"""

from data_utils import TextData
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 读入训练数据
td = TextData()
(x_data, y_data, z_data), (x_labels, y_labels, z_labels) = td.load_data()

vec = CountVectorizer(analyzer='word', max_features=5000,  lowercase = False) # 参数analyzer='word'将剔除单个字？
clf = MultinomialNB()
clf.fit(vec.fit_transform(x_data).toarray(), x_labels)

#joblib.dump(vec, 'vec.m') #保存词空间模型
#joblib.dump(nb_classifier, "clf.m") #保存分类模型

test_data = vec.transform(y_data)
nb_acc = clf.score(test_data.toarray(), y_labels)
print("准确率：", nb_acc)
y_pre = clf.predict(test_data)
print(classification_report(y_labels, y_pre))




















