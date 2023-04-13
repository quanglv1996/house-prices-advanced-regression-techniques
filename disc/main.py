import pandas as pd
import numpy as np
import matplotlib as mlp

from matplotlib import pyplot as plt

path_data = 'D:\house-prices-advanced-regression-techniques\disc\Order_Bundle_Analysis.csv'
df = pd.read_csv(path_data)

df.columns = df.iloc[0]
df = df.drop(df.index[0])

print(df.nunique())


def update_statistical(statistical, list_stat, list_temp):
    # print(statistical)
    if len(statistical) == 0:
        statistical.update({str(list_temp):1})
    else:
        if list_temp in list_stat:
            statistical[str(list_temp)] +=1
        else:
            statistical.update({str(list_temp):1})
    return statistical
            
        

id_uni = df['order-id'].unique()
statistical = {}
list_stat = []

for id in id_uni:
    list_temp = []
    df_product = (df.loc[df['order-id']==id][['name-disc', 'quantity-purchased']]).sort_values(by=['name-disc'])
    
    list_name = []
    for name, count in zip(df_product['name-disc'], df_product['quantity-purchased']):
        if name in list_name:
            temp[name] += int(count)
        else:
            temp = {name:int(count)}
            list_temp.append(temp)
        list_name.append(name)
        
    statistical = update_statistical(statistical, list_stat, list_temp)
    if list_temp not in list_stat:
        list_stat.append(list_temp)
    else:
        pass
    

df = (pd.DataFrame(statistical.items(), columns=['Set', 'Count'])).sort_values(by=['Count'], ascending=False)
print('Total order: {}'.format(df['Count'].sum()))
# df.to_excel('results.xlsx', index=False)


df = df.reset_index()
df_new  = []
for index, row in df.iterrows():
    sum_set = 0
    name_set = row['Set']
    count = row['Count']
    
    new_name_set = name_set.replace('[', '')
    new_name_set = new_name_set.replace(']', '')
    new_name_set = new_name_set.replace('{', '')
    new_name_set = new_name_set.replace('}', '')
    
    list_name = new_name_set.split(',')
    for name in list_name:
        c = name.split(':')[1].strip()
        sum_set += int(c)
        new_name_set = new_name_set.replace('\'', '')
    data = [new_name_set, sum_set, count]
    df_new.append(data)
    
df = pd.DataFrame(df_new, columns = ['Set', 'NPoS', 'Count'])
df.to_excel('results.xlsx', index=False)





