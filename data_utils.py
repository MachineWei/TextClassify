# -*- coding: utf-8 -*-

import re
import jieba
from zhon.hanzi import punctuation
from collections import Counter
from conf import BasePath


class TextData:
    
    def __init__(self, is_seged=False):
        self.is_seged = is_seged
        if not self.is_seged:
            self.local_data()
        self.word2id, self.cat2id = self.build_vocab(BasePath.train_dir, BasePath.vocab_dir)


    def open_file(self, file, mode='r'):
        return open(file, mode, encoding='utf-8', errors='ignore')


    def cut_words(self, segment_func, sentence) -> list:
        """载入分词工具"""
        return segment_func(sentence)


    def to_local(self, input_file, out_file, input_separator='\t'):
        """读取文件数据，并将分词结果写入本地文件"""
        print('将分词结果写入本地ing')
        with self.open_file(input_file) as in_f, self.open_file(out_file,'w') as out_f:
            for line in in_f:
                label, content = line.strip().split(input_separator)
                if content:
                    words = ' '.join(self.cut_words(jieba.lcut, content))
                    out_f.write(label + '\t' + words+'\n')
        print('Done！')


    def read_file(self, input_file, input_separator='\t'):
        """读取文件数据"""
        contents, labels = [], []
        with self.open_file(input_file) as in_f:
            for line in in_f:
                try:
                    label, content = line.strip().split(input_separator)
                    if content:
                        contents.append(content)
                        labels.append(label)
                except:
                    pass
        return contents, labels


    def remove_stopwords(self, string):
        """删除停用词（目前这里仅去标点）"""
        string = re.sub(u"[%s]" % punctuation, " ", string)
        return string.strip()
    
    
    def build_vocab(self, train_segment_dir, vocab_dir, vocab_size=5000):
        """根据训练集构建词汇表并存储"""
        contents, labels = self.read_file(train_segment_dir)
        all_data = []
        for content in contents:
            all_data.extend([c.strip() for c in content.split(' ') if c.strip()!=''])
    
        counter = Counter(all_data)
        count_pairs = counter.most_common(vocab_size - 1)
        words, s = list(zip(*count_pairs))
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        words = ['<PAD>'] + list(words)
        self.open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
        return dict(zip(words, range(len(words)))), dict(zip(set(labels), range(len(set(labels)))))


    def local_data(self):
        self.to_local(BasePath.raw_train, BasePath.train_dir)
        self.to_local(BasePath.raw_test, BasePath.test_dir)
        self.to_local(BasePath.raw_val, BasePath.val_dir)


    def load_data(self):
        """读取文件（已分词）"""
        print('读取已分词文本ing')
        train_data, train_labels=self.read_file(BasePath.train_dir)
        test_data, test_labels=self.read_file(BasePath.test_dir)      
        val_data, val_labels=self.read_file(BasePath.val_dir)
        return (train_data, test_data, val_data), (train_labels, test_labels, val_labels)


    def to_id(self, _data, _labels):
        _data = [[self.word2id.get(word, 0) for word in sen.split(' ')] for sen in _data] 
        _labels = [self.cat2id.get(label) for label in _labels]
        return _data, _labels
       
        
    def load_idata(self):
        """将（已分词）文件转换为id表示"""
        (train_data, test_data, val_data), (train_labels, test_labels, val_labels) = self.load_data()
        print('载入id数据ing')
        x_data, x_labels = self.to_id(train_data, train_labels)
        y_data, y_labels = self.to_id(test_data, test_labels)
        z_data, z_labels = self.to_id(val_data, val_labels)
        return (x_data, y_data, z_data), (x_labels, y_labels, z_labels)
        
        
        
        
