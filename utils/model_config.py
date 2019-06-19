# coding=utf-8
class Config(object):
    def __init__(self, model_name):
        self.model_name = model_name
        # train step
        self.max_features = 20000
        self.maxlen = 100
        self.batch_size = 32
        self.embedding_dims = 50
        self.epochs = 10
        self.class_num = 10
        self.last_activation = 'softmax'
        # han
        self.maxlen_sentence = 10
        self.maxlen_word = 10
