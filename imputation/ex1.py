import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

# pd.set_option('display.max_columns',None)
    
path = '/home/yeonggi/projects/imputation/Data2.csv'

leak = pd.read_csv(path)
leak = leak.drop(['latitude','longitude'], axis = 1)

# def count_empty_cells(col):
#     if col.dtypes == object:
        
#         return (col.str.strip() == "").sum()
    
#     else:
        
#         return col.isna().sum()
    
# empty_counts = df.apply(count_empty_cells)
# print(empty_counts)
# 또는 print(leak.isnull().sum())

# 결측값 개수 찾는 것

# df.drop(df[df['stype'].isna()].index, inplace = True)

# print(df)

# 빈 칸을 포함하는 행을 삭제

leak = leak.dropna()
leak.isna().sum()

# numeric_features = leak.select_dtypes(include = [np.number])


column_find = leak.columns.get_loc('ltype')
print(column_find)

# x = leak.iloc[:,7:26]
# y = leak['ltype']

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)

# print(x_train.shape, y_train_shape, x_test.shape, y_test.shape)

# plt.scatter(x = leak['get1_hz'], y = leak['get2_hz'])
# plt.xlabel('get1_hz', fontsize = 12)
# plt.ylabel('get2_hz', fontsize = 12)

# print(plt.show())

# feature 간에 산점도

# for col in ['get1_hz','get2_hz','get3_hz','get4_hz','get5_hz','get6_hz','get7_hz','get8_hz','get9_hz','get10_hz']:
#     plt.figure(figsize = (10,6))
#     sns.boxplot(x = 'ltype', y = col, hue = 'pdtype', data = leak)
#     plt.title(f'plt of {col} grouped by ltype and ptype')
#     plt.show()
# print(plt.show())

# ltype, pdtype을 기준으로 그룹화하여 boxplot 한 것

# X, y = leak.iloc[:,7:539], leak.iloc[:, ]

    
    











