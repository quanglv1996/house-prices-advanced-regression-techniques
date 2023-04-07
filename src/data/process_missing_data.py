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

# Folder chứa thông tin về bộ dữ liệu MIMIC-III
path_dataset = '/home/quanglv/Downloads/MIMIC-III/'

#Khởi tạo đối tượng Reader cho việc đọc các file csv trong bộ dữ liệu MIMIC-III
reader = Reader(path_dataset=path_dataset)

path_preprocessing = './preprocessing'
path_missing_processing = './missing_processing'
create_folder(path_missing_processing)

'''
-----------------------------------------------------------------------------------------------------
|                               Xử lý missing data                                                  |
-----------------------------------------------------------------------------------------------------
'''

'''
Chuyển missing data sang dạng 24h giờ và fill missing trên cùng 1 features (Sử dụng cho các mô hình LSTM)
'''


def fill_missing_data_timeseries(ratio=50.0, events='chart',using_interpolate= True):
    if events == 'chart':
        file_items = os.path.join(path_preprocessing,'TOP_CHART_ITEMS.csv')  # File chứa thông tin về tỉ lệ missing của các ITEMID
        path = os.path.join(path_preprocessing,'data_timelabel_chartevents')  # Folder chứa thông tin của chartevents ở dạng data with labeltime missing
        if using_interpolate:
            path_train = os.path.join(path_missing_processing,'mice_interpolate/dataset/timeseries_chartevents')  # Folder lưu thông tin của tập training
            create_folder(path_train)
        else:
            path_train = os.path.join(path_missing_processing,'mice/dataset/timeseries_chartevents')  # Folder lưu thông tin của tập training
            create_folder(path_train)
    else:
        file_items = os.path.join(path_preprocessing,'STATISTICAL_MISSING_ITEMS_LABEVENTS.csv')  # File chứa thông tin về tỉ lệ missing của các ITEMID
        path = os.path.join(path_preprocessing,'data_timelabel_labevents')  # Folder chứa thông tin của labevents ở dạng timeseries missing
        if using_interpolate:
            path_train = os.path.join(path_missing_processing,'mice_interpolate/dataset/timeseries_labevents')  # Folder lưu thông tin của tập training
            create_folder(path_train)
        else:
            path_train = os.path.join(path_missing_processing,'mice/dataset/timeseries_labevents')  # Folder lưu thông tin của tập training
            create_folder(path_train)

    # Lấy thông tin và chọn những ITEMID có tỉ lệ missing > ratio (Chuẩn bị cho việc xóa các ITEMID này)
    top_items = pd.read_csv(file_items)
    top_items = top_items[~(top_items['RATIO MISSING'] < ratio)]
    if events == 'chart':
        labelnames = list(top_items.drop_duplicates(subset=['MIMIC LABEL'])['MIMIC LABEL'].values)
    else:
        labelnames = list(map(str, top_items['ITEMID'].values))

    # Tạo dataframe để lưu trữ toàn bộ thông tin của tập training
    data_all = pd.DataFrame()

    # Lấy thông tin của tập training và tập test
    info_dataset = pd.read_csv('DATASET.csv')

    # Tạo training set
    # Chuyển thông tin sang dạng chuỗi thời gian 24 giờ liên tục.
    for i in range(info_dataset.shape[0]):
        print('Progress 1: {}/{}'.format((i + 1), info_dataset.shape[0]))

        icustay_id = (info_dataset.loc[i]).ICUSTAY_ID
        filename = str(icustay_id) + '.csv'
        df = pd.read_csv(os.path.join(path, filename))
        df = df.drop(labelnames, axis=1)  # Xóa bỏ ITEM missing > ratio
        name_columns = df.columns.values[:-1]  # Lấy tên của các cột
        timeseries_fn = []  # List chứa thông tin từng giờ
        for j in range(1, 25):
            temp = df[(df['Time'] < j) & (df['Time'] >= (j - 1))].drop(['Time'], axis=1)
            timeseries_fn.append(temp.mean().tolist())
        data = pd.DataFrame(timeseries_fn, columns=name_columns)

        #Su dung Interpolate
        if using_interpolate:
            data = data.interpolate(limit_direction='both')

        data['ICUSTAY_ID'] = icustay_id
        data_all = data_all.append(data)
    icustay_id = list(data_all['ICUSTAY_ID'].values)
    data_all = data_all.drop(['ICUSTAY_ID'], axis=1)

    # Fill missing data using MICE
    train_cols = list(data_all)
    data_all = pd.DataFrame(IterativeImputer().fit_transform(data_all))
    data_all.columns = train_cols

    data_all.insert(0, 'ICUSTAY_ID', icustay_id, True)
    temps = [frame for season, frame in data_all.groupby(['ICUSTAY_ID'])]
    i = 0
    for temp in temps:
        print('Progress 2: {}/{}'.format((i + 1), len(temps)))
        icustay_id = temp['ICUSTAY_ID'].values[0]
        filename_full = str(icustay_id) + '.csv'
        with open(os.path.join(path_train, filename_full), 'a') as f:
            temp.to_csv(f, encoding='utf-8', header=True)
        i = i + 1

'''
Điền missing data(MICE) cho dữ liệu sử dụng thống k cho XGBoost
'''


def fill_missing_data_statistical(ratio=50.0, events='chart', using_interpolate =  True):
    if events == 'chart':
        file_items = os.path.join(path_preprocessing,'TOP_CHART_ITEMS.csv')  # File chứa thông tin về tỉ lệ missing của các ITEMID
        path = os.path.join(path_preprocessing,'data_timelabel_chartevents')  # Folder chứa thông tin của chartevents ở dạng data with labeltime missing
        if using_interpolate:
            path_train = os.path.join(path_missing_processing,'mice_interpolate/dataset/statistical_chartevents')  # Folder lưu thông tin của tập training
            create_folder(path_train)
        else:
            path_train = os.path.join(path_missing_processing,'mice/dataset/statistical_chartevents')  # Folder lưu thông tin của tập training
            create_folder(path_train)
    else:
        file_items = os.path.join(path_preprocessing,'STATISTICAL_MISSING_ITEMS_LABEVENTS.csv')  # File chứa thông tin về tỉ lệ missing của các ITEMID
        path = os.path.join(path_preprocessing,'data_timelabel_labevents')  # Folder chứa thông tin của labevents ở dạng timeseries missing
        if using_interpolate:
            path_train = os.path.join(path_missing_processing,'mice_interpolate/dataset/statistical_labevents')  # Folder lưu thông tin của tập training
            create_folder(path_train)
        else:
            path_train = os.path.join(path_missing_processing,'mice/dataset/statistical_labevents')  # Folder lưu thông tin của tập training
            create_folder(path_train)


    # Lấy thông tin và chọn những ITEMID có tỉ lệ missing > ratio (Chuẩn bị cho việc xóa các ITEMID này)
    top_items = pd.read_csv(file_items)
    top_items = top_items[~(top_items['RATIO MISSING'] < ratio)]
    if events == 'chart':
        labelnames = list(top_items.drop_duplicates(subset=['MIMIC LABEL'])['MIMIC LABEL'].values)
    else:
        labelnames = list(map(str, top_items['ITEMID'].values))

    # Tạo dataframe để lưu trữ toàn bộ thông tin của tập training
    data_all = pd.DataFrame()

    # Lấy thông tin của tập training và tập test
    info_dataset = pd.read_csv('DATASET.csv')
   
    # Tạo training set
    # Chuyển thông tin sang dạng chuỗi thời gian 24 giờ liên tục.
    for i in range(info_dataset.shape[0]):
        print('Progress 1: {}/{}'.format((i + 1), info_dataset.shape[0]))

        icustay_id = (info_dataset.loc[i]).ICUSTAY_ID
        filename = str(icustay_id) + '.csv'
        df = pd.read_csv(os.path.join(path, filename))
        data = df.drop(labelnames, axis=1)  # Xóa bỏ ITEM missing > ratio
        data = data.interpolate(limit_direction='both')
        data['ICUSTAY_ID'] = icustay_id
        data_all = data_all.append(data)
    icustay_id = list(data_all['ICUSTAY_ID'].values)
    data_all = data_all.drop(['ICUSTAY_ID'], axis=1)

    # Fill missing data using KNN
    train_cols = list(data_all)
    data_all = pd.DataFrame(IterativeImputer().fit_transform(data_all))
    data_all.columns = train_cols

    # Luu de fill missing for test
    data_all.insert(0, 'ICUSTAY_ID', icustay_id, True)
    temps = [frame for season, frame in data_all.groupby(['ICUSTAY_ID'])]
    i = 0
    for temp in temps:
        print('Progress 2: {}/{}'.format((i + 1), len(temps)))
        icustay_id = temp['ICUSTAY_ID'].values[0]
        filename_full = str(icustay_id) + '.csv'
        with open(os.path.join(path_train, filename_full), 'a') as f:
            temp.to_csv(f, encoding='utf-8', header=True)
        i = i + 1
        
def run():

    thread1 = threading.Thread(target=fill_missing_data_timeseries,args=(50.0,'chart', True,))
    thread2 = threading.Thread(target=fill_missing_data_timeseries,args=(50.0,'lab',True,))
    thread3 = threading.Thread(target=fill_missing_data_timeseries,args=(50.0,'chart', False,))
    thread4 = threading.Thread(target=fill_missing_data_timeseries,args=(50.0,'lab',False,))

    thread5 = threading.Thread(target=fill_missing_data_statistical,args=(50.0,'chart', True,))
    thread6 = threading.Thread(target= fill_missing_data_statistical,args=(50.0,'lab',True,))
    thread7 = threading.Thread(target=fill_missing_data_statistical, args=(50.0,'chart',False,))
    thread8 = threading.Thread(target=fill_missing_data_statistical,args=(50.0,'lab',False,))                                    

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