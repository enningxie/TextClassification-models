# coding=utf-8

from keras import Input, Model
from keras.layers import Embedding, Dense, Bidirectional, CuDNNLSTM, TimeDistributed, LSTM
from models.attention import Attention
from keras.preprocessing import sequence


class HAN(object):
    def __init__(self, maxlen_sentence, maxlen_word, max_features, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        # Word part
        input_word = Input(shape=(self.maxlen_word,))
        x_word = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen_word)(input_word)
        x_word = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x_word)  # LSTM or GRU
        # x_word = Bidirectional(LSTM(128, return_sequences=True))(x_word)  # LSTM or GRU
        x_word = Attention(self.maxlen_word)(x_word)
        model_word = Model(input_word, x_word)

        # Sentence part
        input = Input(shape=(self.maxlen_sentence, self.maxlen_word))
        x_sentence = TimeDistributed(model_word)(input)
        x_sentence = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x_sentence)
        # x_sentence = Bidirectional(LSTM(128, return_sequences=True))(x_sentence)  # LSTM or GRU
        x_sentence = Attention(self.maxlen_sentence)(x_sentence)

        output = Dense(self.class_num, activation=self.last_activation)(x_sentence)
        model = Model(inputs=input, outputs=output)
        return model

    def prepare_data(self, x_train, x_test):
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen_sentence * self.maxlen_word)
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen_sentence * self.maxlen_word)
        x_train = x_train.reshape((len(x_train), self.maxlen_sentence, self.maxlen_word))
        x_test = x_test.reshape((len(x_test), self.maxlen_sentence, self.maxlen_word))
        return x_train, x_test
