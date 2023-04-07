import random
import sys
import time
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append('../')

from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from util import results, write_log, draw
from util.util import config_gpu, create_folder, fix_distribution_on_abatchsize
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from model.architecture import *
from model.architecture import Architecture
from util import metrics
from util.reader import Reader
from sklearn.utils import shuffle

path_dataset = '/home/quanglv/Downloads/MIMIC-III/'
# Config GPU
config_gpu(using_config=True,gpu = '0')
reader = Reader(path_dataset=path_dataset)
path_results = './results_kfolds'


# http://digital-thinking.de/keras-how-to-snapshot-your-model-after-x-epochs-based-on-custom-metrics-like-auc/
# Funtion for monitor auroc
def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


# Set parameter for lstm
architecture_name = 'lstm'
# architecture_name = 'gru'

# Parameters 
parameters = {'epochs': 100,
              'batch_size': 256,
              'n_folds': 10,
              'num_units': 32}

# # # Parameters 
# parameters = {'epochs': 80,
#               'batch_size': 256,
#               'n_folds': 10,
#               'num_units': 64}

# # Parameters 
# parameters = {'epochs': 60,
#               'batch_size': 256,
#               'n_folds': 10,
#               'num_units': 128}

# # Parameters 
# parameters = {'epochs': 40,
#               'batch_size': 256,
#               'n_folds': 10,
#               'num_units': 256}

# # Parameters 
# parameters = {'epochs': 30,
#               'batch_size': 256,
#               'n_folds': 10,
#               'num_units': 512}

epochs = parameters['epochs']  # number epochs
batch_size = parameters['batch_size']  # batch size
n_folds = parameters['n_folds']  # number folds
num_units = parameters['num_units']

# Read dataset
X, y = reader.read_data('dataset_lab', dataset='timeseries_mice')




# Get shape input model
timesteps = X.shape[1]
data_dim = X.shape[2]

# Create file name
list_name = [time.strftime("%y%m%d%H%M%S"), architecture_name, str(num_units)]
model_name = '_'.join(list_name)

# Create table contain infor of model
model_info = {
    'Features': ['Time create', 'Architecture', 'Unit numbers','Fold numbers', 'Batch size', 'Epoch numbers', 'Time steps', 'Data dimention'],
    'Value': [time.strftime("%y/%m/%d %H:%M:%S"), architecture_name, num_units,n_folds, batch_size, epochs, timesteps, data_dim]}
model_info = pd.DataFrame.from_dict(model_info)
print(model_info)

# Create folders for save
path_model_results = os.path.join(path_results, model_name)
directory_logs = os.path.join(path_model_results,'log')
create_folder(directory_logs)

directory_model = os.path.join(path_model_results,'model')
create_folder(directory_model)

directory_csv = os.path.join(path_model_results,'csv')
create_folder(directory_csv)

# Save model info
with open(os.path.join(path_model_results,'model_information.csv'), 'w') as f:
    model_info.to_csv(f, encoding='utf-8', header=True, index=False)

# Save tensor board
tensor_board = TensorBoard(log_dir=directory_logs, histogram_freq=0, write_graph=True, write_images=True)

# Reducer Learning rate
lr_reducer = ReduceLROnPlateau(monitor='val_auroc', factor=0.5, patience=3, verbose=1,mode='max')

#EarlyStopping
early_stop = EarlyStopping(monitor='val_auroc', patience=10, verbose=1, mode='max')

# Devide folds using sklear
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random.randint(1,9999))

# Create list to contain history
histories = []
i = 0  # i for count fold

# Create and training model
for train_index, test_index in skf.split(X, y):
    # Split train/test from dataset
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_test,y_test = fix_distribution_on_abatchsize(X_test, y_test, batch_size)

    # Split train val from training set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    X_train, y_train = fix_distribution_on_abatchsize(X_train, y_train, batch_size)
    X_val, y_val = fix_distribution_on_abatchsize(X_val, y_val, batch_size)
    

    # List contain all results for each folds
    results_list = []

    # Set hyperparameter
    fold_n = ['folk' + str(i + 1)]

    # File name contain result each epoch/ fold
    file_name =os.path.join( directory_csv, '_'.join(fold_n) + '.csv')

    # File save model follow Check point
    filepath = os.path.join(directory_model, 'weights_' + '_'.join(fold_n) + '.h5')

    model = None
    architecture = Architecture((timesteps, data_dim))
    # Build model
    if architecture_name == 'lstm':
        print('Build LSTM architecture.')
        model = architecture.lstm_model(lstm_units=num_units,
                                        dense_units=num_units,
                                        bidirectional=True)
    elif architecture_name == 'gru':
        print('Build GRU architecture.')
        model = architecture.gru_model(gru_units=num_units,
                                       dense_units=num_units,
                                       bidirectional=True)
    else:
        print('Not found architecture!')
        sys.exit()

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-7),
                  metrics=['accuracy', auroc])

    print('Training fold ' + str(i + 1))

    # Set checkpoint and monitor AUC_ROC
    checkpoint = ModelCheckpoint(filepath, monitor='val_auroc', verbose=1, save_best_only=True, mode='max')

    # Using class weight
    weight_balance = (y_train[y_train == 1].shape[0]) / y_train.shape[0]

    # Weight balance
    class_weight = {0: weight_balance,
                    1: (1 - weight_balance)}

    # Fit model
    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        verbose=2,
                        class_weight=class_weight,
                        shuffle=False,
                        callbacks=[early_stop,checkpoint, tensor_board, lr_reducer,
                                   results.Results(X_val, y_val, results_list, directory_csv)])
    # Save all history
    histories.append(history)

    # Wirte logs
    write_log.write_log(file_name, results_list)

    # Test best AUCROC model
    print('_____________TEST____________')

    # Load weight model
    model.load_weights(filepath)
    yhat = model.predict(X_test, batch_size=batch_size)
    yhat = np.array(yhat)[:, 0]
    res = metrics.print_metrics_binary(np.array(y_test), yhat, directory_csv, draw_auc=True, fold_i=i)
    res = [res]
    file_result = directory_csv + model_name + '_result.csv'
    write_log.write_log(file_result, res)
    i = i + 1

# Draw chart auc of val and train
with open(os.path.join(path_model_results, 'info_results.pickle'), 'wb') as handle:
        pickle.dump(histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
draw.draw_model_history(n_folds, histories, directory_csv)