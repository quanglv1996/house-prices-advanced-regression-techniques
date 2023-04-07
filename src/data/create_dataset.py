import os
import pickle
import sys
import threading

import numpy as np
import pandas as pd

sys.path.append('../')
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from fancyimpute import IterativeImputer

from util.reader import Reader
from util.util import create_folder

path_create_dataset = './create_dataset'
create_folder(path_create_dataset)
path_missing_processing = './missing_processing'

# Folder chứa thông tin về bộ dữ liệu MIMIC-III
path_dataset = '/home/quanglv/Downloads/MIMIC-III/'

#Khởi tạo đối tượng Reader cho việc đọc các file csv trong bộ dữ liệu MIMIC-III
reader = Reader(path_dataset=path_dataset)

'''
-----------------------------------------------------------------------------------------------------
|                               Tạo dataset training và test  cho LSTM                          |
-----------------------------------------------------------------------------------------------------
'''

def create_dataset_timeseries_events(events='chart', using_iterpolate = True):
    if using_iterpolate:
        dest_path = os.path.join(path_create_dataset, 'timeseries_inter')
        create_folder(dest_path)
        if events == 'chart':
            path_data_train = os.path.join(path_missing_processing,'mice_interpolate/dataset/timeseries_chartevents')
        else:
            path_data_train = os.path.join(path_missing_processing,'mice_interpolate/dataset/timeseries_labevents')
    else:
        dest_path = os.path.join(path_create_dataset, 'timeseries_mice')
        create_folder(dest_path)
        if events == 'chart':
            path_data_train = os.path.join(path_missing_processing,'mice/dataset/timeseries_chartevents')
        else:
            path_data_train = os.path.join(path_missing_processing,'mice/dataset/timeseries_labevents')

    info_dataset =  pd.read_csv('DATASET.csv')

    # Create training set
    X = []
    y = []
    for i in range(info_dataset.shape[0]):
        print('Create dataset: {}/{}'.format(i,info_dataset.shape[0]))
        icustayid_info = info_dataset.loc[i]
        icustayid = int(icustayid_info['ICUSTAY_ID'])
        type = icustayid_info['TYPE']
        age = icustayid_info['AGE']
        gender = icustayid_info['GENDER']
        eth = icustayid_info['ETHNICITY']

        df = pd.read_csv(os.path.join(path_data_train, str(icustayid) + '.csv'))
        df = df.drop(['ICUSTAY_ID', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)

        #Thêm các thông tin như tuổi, giới tính và chủng tộc người
        df['Age'] = int(age)
        if gender == 'M':
            df['Gender'] = 1
        else:
            df['Gender'] = 0
        if eth == 'BLACK/AFRICAN AMERICAN':
            df['Ethnicity'] = 1
        else:
            df['Ethnicity'] = 0
        
        X.append(df.values)
        if type == 1:
            y.append(1)
        else:
            y.append(0)
    X = np.array(X)
    y = np.array(y)
    train_set = {'samples': X, 'labels': y}
    with open(os.path.join(dest_path, 'dataset_' + events + '_all.pickle'), 'wb') as handle:
        pickle.dump(train_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Ratio training: {}/{}'.format(y[y == 1].shape[0], y.shape[0]))


def create_dataset_timeseries_all(using_iterpolate = True):
    if using_iterpolate:
        dest_path = os.path.join(path_create_dataset, 'timeseries_inter')
        create_folder(dest_path)
        path_data_train_chart = os.path.join(path_missing_processing,'mice_interpolate/dataset/timeseries_chartevents')
        path_data_train_lab = os.path.join(path_missing_processing,'mice_interpolate/dataset/timeseries_labevents')
    else:
        dest_path = os.path.join(path_create_dataset, 'timeseries_mice')
        create_folder(dest_path)
        path_data_train_chart = os.path.join(path_missing_processing,'mice/dataset/timeseries_chartevents')
        path_data_train_lab = os.path.join(path_missing_processing,'mice/dataset/timeseries_labevents')

    info_dataset =  pd.read_csv('DATASET.csv')

    # Create training set
    X = []
    y = []
    for i in range(info_dataset.shape[0]):
        print('Create train dataset: {}/{}'.format(i,info_dataset.shape[0]))
        icustayid_info = info_dataset.loc[i]
        icustayid = int(icustayid_info['ICUSTAY_ID'])
        type = icustayid_info['TYPE']
        age = icustayid_info['AGE']
        gender = icustayid_info['GENDER']
        eth = icustayid_info['ETHNICITY']

        df_chart = pd.read_csv(os.path.join(path_data_train_chart, str(icustayid) + '.csv'))
        df_chart = df_chart.drop(['ICUSTAY_ID', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)

        df_lab = pd.read_csv(os.path.join(path_data_train_lab, str(icustayid) + '.csv'))
        df_lab = df_lab.drop(['ICUSTAY_ID', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)

        df = pd.concat([df_chart, df_lab], axis=1)
        df['Age'] = int(age)
        if gender == 'M':
            df['Gender'] = 1
        else:
            df['Gender'] = 0
        if eth == 'BLACK/AFRICAN AMERICAN':
            df['Ethnicity'] = 1
        else:
            df['Ethnicity'] = 0
        X.append(df.values)
        if type == 1:
            y.append(1)
        else:
            y.append(0)
    X = np.array(X)
    y = np.array(y)
    train_set = {'samples': X, 'labels': y}
    with open(os.path.join(dest_path, 'dataset_all.pickle'), 'wb') as handle:
        pickle.dump(train_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Ratio training: {}/{}'.format(y[y == 1].shape[0], y.shape[0]))

'''
-----------------------------------------------------------------------------------------------------
|                               Tạo dataset training và test  cho XGBoost                           |
-----------------------------------------------------------------------------------------------------
'''

'''
Sử dụng 5 biến số thông kê bao gồm min, max, trung bình, độ lệch và phương sai
'''
def statistic_data(X):
    max = np.amax(X, axis=0)
    min = np.amin(X, axis=0)
    avg = np.average(X, axis=0)
    std = np.std(X, axis=0)
    var = np.var(X, axis=0)
    data = np.concatenate((max, min, avg, std, var), axis=0)
    return data


def create_dataset_statistical_events(events='chart',using_iterpolate = True):
    if using_iterpolate:
        dest_path = os.path.join(path_create_dataset, 'statistical_inter')
        create_folder(dest_path)
        if events == 'chart':
            path_data_train = os.path.join(path_missing_processing,'mice_interpolate/dataset/statistical_chartevents')
        else:
            path_data_train = os.path.join(path_missing_processing,'mice_interpolate/dataset/statistical_labevents')
    else:
        dest_path = os.path.join(path_create_dataset, 'statistical_mice')
        create_folder(dest_path)
        if events == 'chart':
            path_data_train = os.path.join(path_missing_processing,'mice/dataset/statistical_chartevents')
        else:
            path_data_train = os.path.join(path_missing_processing,'mice/dataset/statistical_labevents')

    info_dataset = pd.read_csv('DATASET.csv')

    # Create training set
    X = []
    y = []
    for i in range(info_dataset.shape[0]):
        print('Create dataset: {}/{}'.format(i,info_dataset.shape[0]))
        icustayid_info = info_dataset.loc[i]
        icustayid = int(icustayid_info['ICUSTAY_ID'])
        type = icustayid_info['TYPE']
        age = icustayid_info['AGE']
        gender = icustayid_info['GENDER']
        eth = icustayid_info['ETHNICITY']

        df = pd.read_csv(os.path.join(path_data_train, str(icustayid) + '.csv'))
        df = df.drop(['ICUSTAY_ID', 'Unnamed: 0', 'Unnamed: 0.1', 'Time'], axis=1)
        df['Age'] = int(age)
        if gender == 'M':
            df['Gender'] = 1
        else:
            df['Gender'] = 0
        if eth == 'BLACK/AFRICAN AMERICAN':
            df['Ethnicity'] = 1
        else:
            df['Ethnicity'] = 0
        data_statitical = statistic_data(df.values)
        X.append(data_statitical)
        if type == 1:
            y.append(1)
        else:
            y.append(0)
    X = np.array(X)
    y = np.array(y)
    train_set = {'samples': X, 'labels': y}
    with open(os.path.join(dest_path, 'dataset_' + events + '_all.pickle'), 'wb') as handle:
        pickle.dump(train_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Ratio training: {}/{}'.format(y[y == 1].shape[0], y.shape[0]))


def create_dataset_statistical_all(using_iterpolate = True):
    if using_iterpolate:
        dest_path = os.path.join(path_create_dataset, 'statistical_inter')
        create_folder(dest_path)
        path_data_train_chart = os.path.join(path_missing_processing,'mice_interpolate/dataset/statistical_chartevents')
        path_data_train_lab = os.path.join(path_missing_processing,'mice_interpolate/dataset/statistical_labevents')
    else:
        dest_path = os.path.join(path_create_dataset, 'statistical_mice')   
        create_folder(dest_path)
        path_data_train_chart = os.path.join(path_missing_processing,'mice/dataset/statistical_chartevents')
        path_data_train_lab = os.path.join(path_missing_processing,'mice/dataset/statistical_labevents')

    info_dataset = pd.read_csv('DATASET.csv')

    # Create training set
    X = []
    y = []
    for i in range(info_dataset.shape[0]):
        print('Create train dataset: {}/{}'.format(i,info_dataset.shape[0]))
        icustayid_info = info_dataset.loc[i]
        icustayid = int(icustayid_info['ICUSTAY_ID'])
        type = icustayid_info['TYPE']
        age = icustayid_info['AGE']
        gender = icustayid_info['GENDER']
        eth = icustayid_info['ETHNICITY']

        df_chart = pd.read_csv(os.path.join(path_data_train_chart, str(icustayid) + '.csv'))
        df_chart = df_chart.drop(['ICUSTAY_ID', 'Unnamed: 0', 'Unnamed: 0.1','Time'], axis=1)

        df_lab = pd.read_csv(os.path.join(path_data_train_lab, str(icustayid) + '.csv'))
        df_lab = df_lab.drop(['ICUSTAY_ID', 'Unnamed: 0', 'Unnamed: 0.1','Time'], axis=1)

        df = pd.concat([df_chart, df_lab], axis=1)

        df['Age'] = int(age)
        if gender == 'M':
            df['Gender'] = 1
        else:
            df['Gender'] = 0
        if eth == 'BLACK/AFRICAN AMERICAN':
            df['Ethnicity'] = 1
        else:
            df['Ethnicity'] = 0

        data_statitical = statistic_data(df.values)
        X.append(data_statitical)
        if type == 1:
            y.append(1)
        else:
            y.append(0)
    X = np.array(X)
    y = np.array(y)
    train_set = {'samples': X, 'labels': y}
    with open(os.path.join(dest_path, 'dataset_all.pickle'), 'wb') as handle:
        pickle.dump(train_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Ratio training: {}/{}'.format(y[y == 1].shape[0], y.shape[0]))


    
def run():

    thread1 = threading.Thread(target=create_dataset_timeseries_events,args=('lab', True,))
    thread2 = threading.Thread(target=create_dataset_timeseries_all,args=(True,))
    thread3 = threading.Thread(target=create_dataset_timeseries_events,args=('lab', False,))
    thread4 = threading.Thread(target=create_dataset_timeseries_all,args=(False,))
    thread5 = threading.Thread(target=create_dataset_statistical_events,args=('lab', True,))
    thread6 = threading.Thread(target= create_dataset_statistical_all,args=(True,))
    thread7 = threading.Thread(target=create_dataset_statistical_events, args=('lab',False,))
    thread8 = threading.Thread(target=create_dataset_statistical_all,args=(False,))                                    

    # Bat dau
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    thread7.start()
    thread8.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()
    thread7.join()
    thread8.join()

    print('Processing completed!')
    print('')
    
run()