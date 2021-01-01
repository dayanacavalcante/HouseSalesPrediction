# Imports
import pandas as pd
from sklearn import linear_model as lm
import numpy as np
from sklearn import model_selection as ms 
from sklearn import metrics as m 

# Loading Data
data = pd.read_csv('C:\\Users\\RenanSardinha\\Documents\\Data Science\\House Prices\\Data\\kc_house_data.csv')
print("==============================  DATA  =======================================")
print(data)
print("================================  END DATA  =====================================")

# Data Preparation
print("==============================  DATA SUM IS NULL  =======================================")
print(data.isnull().sum())
print("================================  END DATA SUM IS NULL  =====================================")

print("==============================  DATA UNIQUE  =======================================")
print (data.apply(lambda x: len (x.unique())))
print("================================  END DATA UNIQUE  =====================================")

print("==============================  DATA 'ID' VALUE_COUNTS  =======================================")
print(data['id'].value_counts())
print("==============================  END DATA 'ID' VALUE_COUNTS  =======================================")

print("==============================  DATA 'ID' SUM VALUE_COUNTS  =======================================")
print(data['id'].value_counts().sum())
print("============================== END DATA 'ID' SUM VALUE_COUNTS  =======================================")

print("==============================  DATA 'PRICE' MEAN  =======================================")
print(data['price'].mean())
print("==============================  END DATA 'PRICE' MEAN  =======================================")

print("==============================  X = DATA DROP 'DATE' AND 'PRICE'  =======================================")
x = data.drop(['date','price'], axis=1)
print(x)
print("============================== END X  =======================================")

print("==============================  Y = DATA 'PRICE'  =======================================")
y = data['price']
print(y)
print("==============================  END Y  =======================================")


x_train, x_test, y_train, y_test = ms.train_test_split(x,y,test_size=.2,random_state=1)

print("==============================  X_TRAIN  =======================================")
x_train = data.drop(['date', 'price'], axis=1)
print(x_train)
print("==============================  END X_TRAIN  =======================================")

print("==============================  Y_TRAIN  =======================================")
y_train = data['price']
print(y_train)
print("==============================  END Y_TRAIN  =======================================")

# Model Training
model_lr = lm.LinearRegression()
model_lr.fit(x_train, y_train)
pred = model_lr.predict(x_train)

print("==============================  PRED[O:100]  =======================================")
print(pred[0:100])
print("==============================  END PRED[0:100]  =======================================")

# Prediction-training
pred_train = model_lr.predict(x_train)
#Predicition-test
pred_test = model_lr.predict(x_test)
# Performance Metrics

data1 = data.copy()
data1['prediction'] = pred
print("==============================  HEAD HOUSE PRICE FORECAST  =======================================")
print(data1[['id', 'price', 'prediction']].head())
print("==============================  END HEAD HOUSE PRICE FORECAST  =======================================")

data1['error'] = data1['price'] - data1['prediction']
print(data1[['id', 'price', 'prediction', 'error']].head())
data1['error_abs'] = np.abs(data1['error'])
print(data1[['id', 'price', 'prediction', 'error', 'error_abs']].head())

print("==============================  AVERAGE ERROR  =======================================")
print(np.sum(data1['error_abs']) / len(data1['error_abs']))
print("==============================  END AVERAGE ERROR  =======================================")

# Mean Absolute Error
mae = np.mean(data1['error_abs'])
print("==============================  MAE  =======================================")
print('MAE:{}'.format(mae))
print("==============================  END MAE  =======================================")

data1['error_perc'] = ((data1['price'] - data1['prediction']) / data1['price'])
print(data1[['id', 'price', 'prediction', 'error', 'error_abs', 'error_perc']].head())

data1['error_perc_abs'] = np.abs(data1['error_perc'])
print(data1[['id', 'price', 'prediction', 'error', 'error_abs', 'error_perc', 'error_perc_abs']].head())

# Mean Absolute Percent Error
mape = np.mean(data1['error_perc_abs'])
print("==============================  MAPE  =======================================")
print( 'MAPE: {}'.format(mape))
print("==============================  END MAPE  =======================================")

# Training - MAE, MAPE
mae_train = m.mean_absolute_error(y_train, pred_train)
print(mae_train)
mape_train = np.mean(np.abs((y_train - pred_train) / y_train))
print(mape_train)

# Test
mae_test = m.mean_absolute_error(y_test, pred_test)
mape_test = np.mean(np.abs((y_test - pred_test) / y_test))

df = {
'Dataframe': ['training', 'test'],
'MAE': [mae_train, mae_test],
'MAPE': [mape_train, mape_test]}
print(pd.DataFrame(df))