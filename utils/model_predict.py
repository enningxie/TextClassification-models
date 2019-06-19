# coding=utf-8
import pickle
from keras.preprocessing import sequence
from keras.models import load_model
import numpy as np

CHAR2ID_PATH = '/Users/xieenning/Documents/Codes/text_classification/TextClassification-models/char2id.pickle'
INDEX2WORD_PATH = '/Users/xieenning/Documents/Codes/text_classification/TextClassification-models/index2word.pickle'
MODEL_PATH = '/Users/xieenning/Documents/Codes/text_classification/TextClassification-models/text_cnn_0.73.h5'


class Predicter(object):
    def __init__(self, raw_sentence):
        self.raw_data = raw_sentence

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
    def _string2id(self, s, char2id):
        return [char2id.get(i, 1) for i in s]

    def _load_data(self, data_path):
        with open(data_path, 'rb') as file:
            restored_data = pickle.load(file)
        return restored_data

    def predict(self):
        model = load_model(MODEL_PATH)
        tmp_sent = self.raw_data
        tmp_sent = self._strQ2B(tmp_sent)
        tmp_sent = tmp_sent.lower()
        char2id = self._load_data(CHAR2ID_PATH)
        tmp_sent = self._string2id(tmp_sent, char2id)
        tmp_sent = np.asarray(tmp_sent).reshape((1, -1))
        tmp_sent = sequence.pad_sequences(tmp_sent, maxlen=100)
        result_index = np.argmax(model.predict(tmp_sent))
        index2word = self._load_data(INDEX2WORD_PATH)
        return index2word[result_index]


if __name__ == '__main__':
    raw_str = '摩雷的听爱卓和优特声听人声都非常OK，而且价格不算高。DLS的低频应该是不用担心的，建议后门装，不加低音的前提下！可以加好友聊配置方案！'
    raw_str2 = '长城配置好！性价比高！'
    print(Predicter(raw_str2).predict())


