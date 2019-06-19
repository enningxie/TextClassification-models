# coding=utf-8
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle


class DataLoader(object):
    def __init__(self):
        # 字典中保留字最少出现次数
        self.min_count = 5
        self.data_path = '/Users/xieenning/Documents/Codes/text_classification/TextClassification-models/' \
            'data/car_text_classification.csv'
        self._process_data()

    # 全角转半角
    def _strQ2B(self, ustring):
        rstring = ''
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        return rstring

    # 将字符串转换为数字序列
    def _string2id(self, s):
        return [self.char2id.get(i, 1) for i in s]

    # 将数字序列转换为字符串
    def _id2string(self, ids):
        return ''.join([self.id2char.get(i, '<UNK>') for i in ids])

    def _process_data(self):
        raw_data = pd.read_csv(self.data_path, encoding='utf-8', header=None)

        # raw_data column 0 for feature, column 1 for label
        # process feature start
        raw_data[0] = raw_data[0].apply(self._strQ2B)
        raw_data[0] = raw_data[0].str.lower()
        chars = {}
        for s in tqdm(iter(raw_data[0])):
            for c in s:
                if c not in chars:
                    chars[c] = 0
                chars[c] += 1

        # 0: padding标记
        # 1: unk标记
        self.chars = {i: j for i, j in chars.items() if j >= self.min_count}
        self.id2char = {i + 2: j for i, j in enumerate(chars)}
        self.char2id = {j: i for i, j in self.id2char.items()}
        with open('./char2id.pickle', 'wb') as f:
            pickle.dump(self.char2id, f, -1)
        self.X = raw_data[0].apply(self._string2id).values
        # # maxlen = 92
        # a = raw_data[2].apply(len)
        # print(a.mean())
        # print(a.max())
        # print(np.percentile(a, 90))
        # process feature end
        # process label start
        word2index = {word: i for i, word in enumerate(raw_data[1].unique())}
        index2word = {i: word for word, i in word2index.items()}
        # 保存数据
        with open('./index2word.pickle', 'wb') as f:
            pickle.dump(index2word, f, -1)
        self.y = raw_data[1].apply(word2index.get).values
        # process label end

    def load_data_kfold(self, n_splits=10, shuffle=True, random_state=42):
        folds = list(
            StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(self.X, self.y))
        return folds

    def load_data_normal(self, shuffle=True, test_size=0.2):
        # process data for training
        x_total = self.X
        y_total = self.y
        if shuffle:
            index_array = np.arange(len(self.X))
            np.random.shuffle(index_array)
            x_total = x_total[index_array]
            y_total = y_total[index_array]
        x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=test_size, random_state=42)
        return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data_normal()

    print(data_loader.X.shape)
    print(data_loader.X[:5])
    print(data_loader.y.shape)
    print(data_loader.y[:5])
    print(x_train.shape)
    print(x_test.shape)
