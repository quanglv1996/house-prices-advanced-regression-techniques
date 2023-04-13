import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


class PreProcessing(object):
    def __init__(self):
        self.fill_missing_number_dic ={}
        self.fill_missing_cate_dic = {}
        self.onehot_encoder = OneHotEncoder()

    def load_data(self, path_csv):
        df = pd.read_csv(path_csv)
        return df
    
    '''
    Steps processing data
    '''
    def process_missing_data_number(self, df, update_fill_missing=False):
        df_number = df.select_dtypes(include='number')
        for col in df_number:
            mean = df_number[col].mean()
            if update_fill_missing:
                self.fill_missing_number_dic.update({col:mean})
                df_number[col] = df_number[col].replace(np.nan, mean)
            else:
                df_number[col] = df_number[col].replace(np.nan, self.fill_missing_number_dic[col])
        
        return df_number
    
    def process_missing_data_categorical(self, df, update_fill_missing=False):
        df_ca = df.select_dtypes(exclude='number')
        for col in df_ca:
            values = df_ca[col].mode().values
            if update_fill_missing:
                print('sdasdsd', type(col), type(values))
                self.fill_missing_cate_dic.update({col: values})

                df_ca[col].fillna(value=values[0], inplace=True)
            else:
                print('vvvvvvvvvv', type(self.fill_missing_cate_dic[col]))
                df_ca[col].fillna(value=self.fill_missing_cate_dic[col], inplace=True)
                
        # print(df_ca)
        
        for i, col in enumerate(df_ca):
            count = df_ca[col].isna().sum()
            if count > 0:
                print('Col: {}| Count: {}'.format(col, count))
        
        
        
        return df_ca
    
    def process_missing(self, df, update_fill_missing=False):
        # df_number = self.process_missing_data_number(df, update_fill_missing)
        df_cate = self.process_missing_data_categorical(df, update_fill_missing)
        # for i, col in enumerate(df_cate):
        #     count = df_cate[col].isna().sum()
        #     if count > 0:
        #         print('Col: {}| Count: {}'.format(col, count))
        
        df_cate_onehot = self.create_one_hot_vector(df_cate, update_categorical=update_fill_missing)
        # print(df_cate_onehot)
        
        # df_data = df_number.join(df_cate_onehot)
        # df_data.drop(['Grvl'], axis=1, inplace=True)
        
        # return df_data
        
    def create_one_hot_vector(self, df_categorical, update_categorical=False):
        enc_data = None
        if update_categorical:
            for col in df_categorical:
                df_categorical[col] = df_categorical[col].astype('category')
            enc_data=pd.DataFrame(self.onehot_encoder.fit_transform(df_categorical).toarray())
        else:
            # for col in df_categorical:
            #     df_categorical[col] = df_categorical[col].astype('category')
            enc_data=pd.DataFrame(self.onehot_encoder.transform(df_categorical).toarray())
        
        print(enc_data)

  
    
def main():
    path_data_train = 'D:\house-prices-advanced-regression-techniques\data/train.csv'
    path_data_test = 'D:\house-prices-advanced-regression-techniques\data/test.csv'
    
    preprocess = PreProcessing()
    
    # Loading data
    df_train = preprocess.load_data(path_csv=path_data_train)
    X_train = df_train.loc[:, df_train.columns != 'SalePrice']
    Y_train = df_train['SalePrice']
    
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)
    
    # Process missing data
    x_train = preprocess.process_missing(X_train, update_fill_missing=True)
    
    ccc = preprocess.process_missing(x_val, update_fill_missing=False)
    
    
    # print(x_train)
    # x_val = preprocess.process_missng(x_val, update_fill_missing=False)
    # print(x_val)

if __name__ == '__main__':
    main()
    
    
    
    
    