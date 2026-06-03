import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


df = pd.read_csv('../03_telco-churn/telco-churn.csv')


df.columns = df.columns.str.lower()
categorical_columns = df.select_dtypes(include='string').columns.to_list()
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.str.lower().str.replace(' ', '_', regex=False))
df['totalcharges'] = df['totalcharges'].apply(pd.to_numeric, errors='coerce')
df['seniorcitizen'] = df['seniorcitizen'].astype('str')
df['totalcharges'] = df['totalcharges'].fillna(0)
df['churn'] = (df.churn == 'yes').astype('int')

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)

y_train = df_train.churn
y_val = df_val.churn
y_test = df_test.churn

del df_train['churn']
del df_val['churn']
del df_test['churn']


numerical_columns = df_train_full.columns[(df_train_full.dtypes=='float') | (df_train_full.dtypes=='int')].tolist()
numerical_columns.remove('churn')

categorical_columns = df_train_full.columns[df_train_full.dtypes=='str'].tolist()
categorical_columns.remove('customerid')


y_train_full = df_train_full.churn.values
df_train_one_hot_encoded = pd.get_dummies(df_train_full[categorical_columns], dtype=int)
df_train_full = pd.concat([df_train_full[numerical_columns], df_train_one_hot_encoded], axis=1)

df_train_one_hot_encoded = pd.get_dummies(df_train[categorical_columns], dtype=int)
df_train = pd.concat([df_train[numerical_columns], df_train_one_hot_encoded], axis=1)

df_val_one_hot_encoded = pd.get_dummies(df_val[categorical_columns], dtype=int)
df_val = pd.concat([df_val[numerical_columns], df_val_one_hot_encoded], axis=1)

df_test_one_hot_encoded = pd.get_dummies(df_test[categorical_columns], dtype=int)
df_test = pd.concat([df_test[numerical_columns], df_test_one_hot_encoded], axis=1)

model = LogisticRegression(max_iter=100)
model.fit(df_train_full, y_train_full)

y_pred = model.predict(df_val)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_val, y_pred))

import pickle

with open('model.bin', 'wb') as file:
    pickle.dump(model, file)