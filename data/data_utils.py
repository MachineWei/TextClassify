# -*- coding: utf-8 -*-

import re
import jieba
from collections import Counter


class DataPath:
    raw_train = './data/raw_data/cnews.train.txt'
    raw_test = './data/raw_data/cnews.test.txt'
    raw_val = './data/raw_data/cnews.val.txt'

    train_dir = './data/segment_data/train.txt'
    test_dir = './data/segment_data/test.txt'
    val_dir = './data/segment_data/val.txt'
    vocab_dir = './ckpts/vocab.txt'
    target_dir = './ckpts/target.txt'
    stop_words = './ckpts/stop_words.txt'


class LoadData:
    
    re_han = re.compile("([\u4E00-\u9FD5]+)")
    
    def __init__(self, features='word'):
        """
        features: word or char, 词向量或字向量
        """
        self.features = features
        self.stop_words = self.open_file(DataPath.stop_words).read().strip().split('\n')
        # self.to_local()
        self.word2id, self.cat2id = self.build_vocab(DataPath.train_dir, DataPath.vocab_dir, DataPath.target_dir)
    
    def open_file(self, file, mode='r'):
        return open(file, mode, encoding='utf-8', errors='ignore')

    def cut_words(self, sentence : str) -> list:
        if self.features == 'word':
            return jieba.lcut(sentence)
        else:
            return list(sentence)

    def remove_stopwords(self, words):
        """
        删除停用词
        """
        return [word for word in words if word not in self.stop_words]

    def text_progress(self, content):
        """
        文本预处理：分词、去停用词
        """
        sentences = self.re_han.findall(content)
        ch_words = [' '.join(self.remove_stopwords(self.cut_words(sentence))) for sentence in sentences]
        return ' '.join(ch_words)

    def __to_local(self, input_file, out_file):
        """
        经文本预处理结果写入本地
        """
        with self.open_file(input_file) as in_f, self.open_file(out_file,'w') as out_f:
            for line in in_f:
                label, content = line.strip().split('\t')
                if content:
                    words = self.text_progress(content)
                    out_f.write(label + '\t' + words+'\n')

    def to_local(self):
        raw_file = [DataPath.raw_train, DataPath.raw_test, DataPath.raw_val]
        progressed_file = [DataPath.train_dir, DataPath.test_dir, DataPath.val_dir]
        for input_file, out_file in zip(raw_file,progressed_file):
            self.__to_local(input_file, out_file)

    def read_file(self, input_file):
        """
        读取处理后的本地数据
        """
        contents, labels = [], []
        with self.open_file(input_file) as in_f:
            for i, line in enumerate(in_f):
                try:
                    label, content = line.strip().split('\t')
                    if content:
                        contents.append(content)
                        labels.append(label)
                except:
                    print("第%d行发现错误" % i)
                    pass
        return contents, labels

    def build_vocab(self, train_segment_dir, vocab_dir, target_dir, vocab_size=5000):
        """
        根据训练集构建词汇表并存储
        """
        contents, labels = self.read_file(train_segment_dir)
        labels = sorted(list(set(labels)))
        all_data = []
        for content in contents:
            all_data.extend([c.strip() for c in content.split(' ') if c.strip() != ''])

        counter = Counter(all_data)
        count_pairs = counter.most_common(vocab_size - 2)
        words, s = list(zip(*count_pairs))

        words = ['<PAD>', '<UNK>'] + list(words)         # pad映射到0，unk映射到1
        self.open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
        self.open_file(target_dir, mode='w').write('\n'.join(labels) + '\n')
        return dict(zip(words, range(len(words)))), dict(zip(labels, range(len(labels))))

    def to_id(self, _data, _labels):
        """
        转换为id表示
        """
        _data = [[self.word2id.get(word, self.word2id.get('<UNK>')) for word in sen.split(' ')] for sen in _data]
        _labels = [self.cat2id.get(label) for label in _labels]
        return _data, _labels

    def pad_sequences(self, _data_id, sequence_length):
        for i, d in enumerate(_data_id):
            if len(d) < sequence_length:
                _data_id[i] = d + [0] * (sequence_length - len(d))
            else:
                _data_id[i] = d[:sequence_length]
        return _data_id

    def load_data(self, name='train', sequence_length=100):
        if name=='train':
            x_data, x_labels = self.read_file(DataPath.train_dir)
            x_data, x_labels = self.to_id(x_data, x_labels)
            x_data = self.pad_sequences(x_data, sequence_length)
            return x_data, x_labels
        elif name=='val':
            z_data, z_labels = self.read_file(DataPath.val_dir)
            z_data, z_labels = self.to_id(z_data, z_labels)
            z_data = self.pad_sequences(z_data, sequence_length)            
            return z_data, z_labels        
        else:
            y_data, y_labels = self.read_file(DataPath.test_dir)
            y_data, y_labels = self.to_id(y_data, y_labels)
            y_data = self.pad_sequences(y_data, sequence_length)               
            return y_data, y_labels


