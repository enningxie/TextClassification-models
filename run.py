# coding=utf-8

from models.fast_text import FastText
from models.han import HAN
from models.rcnn import RCNN
from models.rcnn_variant import RCNNVariant
from models.text_att_birnn import TextAttBiRNN
from models.text_birnn import TextBiRNN
from models.text_cnn import TextCNN
from models.text_rnn import TextRNN
from utils.model_config import Config
from utils.model_train import Trainer

MODEL_CONFIG = Config('text_classifier')

NAME2MODEL = {'fast_text': FastText(MODEL_CONFIG.maxlen,
                                    MODEL_CONFIG.max_features,
                                    MODEL_CONFIG.embedding_dims,
                                    MODEL_CONFIG.class_num,
                                    MODEL_CONFIG.last_activation),
              'han': HAN(MODEL_CONFIG.maxlen_sentence,
                         MODEL_CONFIG.maxlen_word,
                         MODEL_CONFIG.max_features,
                         MODEL_CONFIG.embedding_dims,
                         MODEL_CONFIG.class_num,
                         MODEL_CONFIG.last_activation),
              'rcnn': RCNN(MODEL_CONFIG.maxlen,
                           MODEL_CONFIG.max_features,
                           MODEL_CONFIG.embedding_dims,
                           MODEL_CONFIG.class_num,
                           MODEL_CONFIG.last_activation),
              'rcnn_variant': RCNNVariant(MODEL_CONFIG.maxlen,
                                          MODEL_CONFIG.max_features,
                                          MODEL_CONFIG.embedding_dims,
                                          MODEL_CONFIG.class_num,
                                          MODEL_CONFIG.last_activation),
              'text_att_birnn': TextAttBiRNN(MODEL_CONFIG.maxlen,
                                             MODEL_CONFIG.max_features,
                                             MODEL_CONFIG.embedding_dims,
                                             MODEL_CONFIG.class_num,
                                             MODEL_CONFIG.last_activation),
              'text_birnn': TextBiRNN(MODEL_CONFIG.maxlen,
                                      MODEL_CONFIG.max_features,
                                      MODEL_CONFIG.embedding_dims,
                                      MODEL_CONFIG.class_num,
                                      MODEL_CONFIG.last_activation),
              'text_cnn': TextCNN(MODEL_CONFIG.maxlen,
                                  MODEL_CONFIG.max_features,
                                  MODEL_CONFIG.embedding_dims,
                                  MODEL_CONFIG.class_num,
                                  MODEL_CONFIG.last_activation),
              'text_rnn': TextRNN(MODEL_CONFIG.maxlen,
                                  MODEL_CONFIG.max_features,
                                  MODEL_CONFIG.embedding_dims,
                                  MODEL_CONFIG.class_num,
                                  MODEL_CONFIG.last_activation)}


def main(model_name, cross_validation=True):
    model = NAME2MODEL.get(model_name, None)
    if not model:
        print("We have no model named '{}'.".format(model_name))
        return
    model_config = Config(model_name)
    model_trainer = Trainer(model_config, model)
    if cross_validation:
        model_trainer.train_kflods()
        return
    model_trainer.train_normal()


if __name__ == '__main__':
    main('text_rnn', False)
