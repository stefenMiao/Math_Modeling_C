#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data =pd.read_csv("/Users/steafen/Desktop/data2.csv")
x_data=data.drop(["Type","color","Wenshi"],axis=1)
y_data=data.iloc[:,-3]


model=LogisticRegression()
model.fit(x_data,y_data)
print(model.intercept_)
print(model.coef_)

x_data.iloc[10].values

test=x_data.iloc[10].values.reshape(1,-1)
print(model.predict(test)[0])

# 进行预测
data_test =pd.read_csv("/Users/steafen/Desktop/data3.csv",header=None)

fenghua=data_test.loc[:,0]
print(fenghua)
#pd.concat(['data_test','fenghua'],axis=1)
data_test.drop(0,axis=1,inplace=True)
data_test=pd.concat([data_test,fenghua],axis=1)
test_x=data_test.loc[7].values.reshape(1,-1)

print(model.predict(test_x)[0])


accuracy_score(model.predict(x_data),y_data) # 计算在训练集上的准确度

mse = np.mean((model.predict(x_data) - y_data) ** 2)

coef=model.coef_
