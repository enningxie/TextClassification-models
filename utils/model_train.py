# coding=utf-8
from utils.data_loader import DataLoader
from keras.callbacks import EarlyStopping
import numpy as np


class Trainer(object):
    def __init__(self, config_obj, model_obj, optimizer='adam', loss_func='sparse_categorical_crossentropy'):
        self.model_obj = model_obj
        self.model_config = config_obj
        self.data_loader = DataLoader()
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')

    def train_normal(self):
        # prepare input data
        (x_train, y_train), (x_test, y_test) = self.data_loader.load_data_normal()
        x_train, x_test = self.model_obj.prepare_data(x_train, x_test)
        model = self.model_obj.get_model()
        model.compile(self.optimizer, self.loss_func, metrics=['accuracy'])
        print('Training...')
        history = model.fit(x_train, y_train,
                            batch_size=self.model_config.batch_size,
                            epochs=self.model_config.epochs,
                            callbacks=[self.early_stopping],
                            validation_data=(x_test, y_test))
        print('Evaluating...')
        val_result = model.evaluate(x_test, y_test)
        print("{}'s acc: {};".format(self.model_config.model_name, val_result[1]))
        print("{}'s loss: {}.".format(self.model_config.model_name, val_result[0]))

    def train_kflods(self):
        folds = self.data_loader.load_data_kfold()
        x = self.data_loader.X
        y = self.data_loader.y
        val_loss_flods = []
        val_acc_flods = []
        best_acc = 0
        for j, (train_idx, val_idx) in enumerate(folds):
            print('\nFold ', j + 1)
            x_train_cv = x[train_idx]
            y_train_cv = y[train_idx]
            x_valid_cv = x[val_idx]
            y_valid_cv = y[val_idx]
            x_train_cv, x_valid_cv = self.model_obj.prepare_data(x_train_cv, x_valid_cv)
            model = self.model_obj.get_model()
            model.compile(self.optimizer, self.loss_func, metrics=['accuracy'])
            print('Training...')
            history = model.fit(x_train_cv, y_train_cv,
                                batch_size=self.model_config.batch_size,
                                epochs=self.model_config.epochs,
                                callbacks=[self.early_stopping],
                                validation_data=(x_valid_cv, y_valid_cv))
            print('Evaluate...')
            # result = model.predict(x_test)
            val_result = model.evaluate(x_valid_cv, y_valid_cv)
            print("[{}]: {}'s acc: {};".format(j + 1, self.model_config.model_name, val_result[1]))
            print("[{}]: {}'s loss: {}.".format(j + 1, self.model_config.model_name, val_result[0]))
            if val_result[1] > best_acc:
                best_acc = val_result[1]
                model.save('./{}_{:.2f}.h5'.format(self.model_config.model_name, best_acc))
            val_loss_flods.append(val_result[0])
            val_acc_flods.append(val_result[1])
        sorted_index = np.argsort(val_acc_flods)[::-1]
        print("{}'s val_acc: {}".format(self.model_config.model_name, np.asarray(val_acc_flods)[sorted_index]))
        print("{}'s val_loss: {}".format(self.model_config.model_name, np.asarray(val_loss_flods)[sorted_index]))
