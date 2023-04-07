import sys
import random

sys.path.append('../')
import os
import threading
from sklearn.utils import shuffle
import keras
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

'''
Config GPU 
'''


def config_gpu(using_config=True, gpu = '1'):
    if using_config:
        print('Using config GPU!')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        # Config minimize GPU with model
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)
    else:
        print('Not using config GPU!')


'''
Create folder if not exist
'''


def create_folder(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs((path_folder))
        print('Directory {} created successfully!'.format(path_folder))
    else:
        print('Directory {} already exists!'.format(path_folder))


'''
Write data to csv file
'''


def write_csv(data, path_file):
    if not os.path.exists(path_file):
        with open(path_file, 'w') as f:
            data.to_csv(f, encoding='utf-8', header=True, index=False)


'''
Process scale data in batch
'''
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def fix_distribution_on_abatchsize(X,y, batch_size):
    X, y = shuffle(X, y)

    index_1 = list(np.where(y==1)[0])
    index_0 = list(np.where(y==0)[0])

    num_inter = int((X.shape[0]/batch_size))*2
   
    list1 = list(split(index_0, num_inter))
    list0 = list(split(index_1, num_inter))
    res = [x0+x1 for x1, x0 in zip(list1, list0)]
    for sublist in res:
        random.shuffle(sublist) 

    flat_list = [item for sublist in res for item in sublist]
    X = X[flat_list]
    y = y[flat_list]
    return X, y
