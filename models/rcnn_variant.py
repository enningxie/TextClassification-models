# coding=utf-8

from keras import Input, Model
from keras.layers import Embedding, Dense, Concatenate, Conv1D, Bidirectional, CuDNNLSTM, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, LSTM
from keras.preprocessing import sequence


class RCNNVariant(object):
    """Variant of RCNN.

        Base on structure of RCNN, we do some improvement:
        1. Ignore the shift for left/right context.
        2. Use Bidirectional LSTM/GRU to encode context.
        3. Use Multi-CNN to represent the semantic vectors.
        4. Use ReLU instead of Tanh.
        5. Use both AveragePooling and MaxPooling.
    """

    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input = Input((self.maxlen,))

        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input)

        x_context = Bidirectional(CuDNNLSTM(128, return_sequences=True))(embedding)
        # x_context = Bidirectional(LSTM(128, return_sequences=True))(embedding)
        x = Concatenate()([embedding, x_context])

        convs = []
        for kernel_size in range(1, 5):
            conv = Conv1D(128, kernel_size, activation='relu')(x)
            convs.append(conv)
        poolings = [GlobalAveragePooling1D()(conv) for conv in convs] + [GlobalMaxPooling1D()(conv) for conv in convs]
        x = Concatenate()(poolings)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model

    def prepare_data(self, x_train, x_test):
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        return x_train, x_test
