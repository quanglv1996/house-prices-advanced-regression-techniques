import sys

sys.path.append('../')

import numpy as np
from util import metrics

import keras


class Results(keras.callbacks.Callback):
    def __init__(self, val_X, val_y, results, save_image):
        super(Results, self).__init__()
        self.val_X = val_X
        self.val_y = val_y
        self.batch_size = val_X.shape[0]
        self.results = results
        self.save_image = save_image

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on validation")
        yhat = self.model.predict(self.val_X, batch_size=self.batch_size)
        yhat = np.array(yhat)[:, 0]
        result = metrics.print_metrics_binary(np.array(self.val_y), yhat, save_image=self.save_image)
        self.results.append(result)
