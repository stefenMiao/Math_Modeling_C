#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import statsmodels.formula.api as sm

data =pd.read_csv("/Users/steafen/Desktop/data2.csv")
x_data=data.iloc[:,-4:]
x_data

# 获得哑变量
dummies_wenshi=pd.get_dummies(x_data['Wenshi'],prefix='Wenshi')
dummies_type=pd.get_dummies(x_data['Type'],prefix='Type')
dummies_color=pd.get_dummies(x_data['color'],prefix='color')
dummies_fenghua=pd.get_dummies(x_data['Fenghua'],prefix='Fenghua')

# 舍弃一个虚拟变量，保证得到满秩矩阵
dummies_wenshi.drop(columns=['Wenshi_3'],inplace=True)
dummies_type.drop(columns=['Type_1'],inplace=True)
dummies_color.drop(columns=['color_8'],inplace=True)
dummies_fenghua.drop(columns=['Fenghua_1'],inplace=True)
dum_data=pd.concat(objs=[dummies_color,dummies_fenghua,dummies_type,dummies_wenshi],axis='columns')

y_data=data.iloc[:,0]
result=pd.concat([dum_data,y_data],axis='columns')

model=sm.ols(formula='SiO2~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=result).fit()
model.summary()

alldata=pd.concat([dum_data,data],axis=1)

model_na2o=sm.ols(formula='Na2O~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_na2o.summary()

model_k2o=sm.ols(formula='K2O~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_k2o.summary()

model_CaO=sm.ols(formula='CaO~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_CaO.summary()

model_MgO=sm.ols(formula='MgO~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_MgO.summary()

model_Al2O3=sm.ols(formula='Al2O3~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_Al2O3.summary()

model_Fe2O3=sm.ols(formula='Fe2O3~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_Fe2O3.summary()

model_CuO=sm.ols(formula='CuO~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_CuO.summary()

model_PbO=sm.ols(formula='PbO~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_PbO.summary()

model_BaO=sm.ols(formula='BaO~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_BaO.summary()

model_P2O5=sm.ols(formula='P2O5~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_P2O5.summary()

model_SrO=sm.ols(formula='SrO~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_SrO.summary()

model_SnO2=sm.ols(formula='SnO2~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_SnO2.summary()

model_SO2=sm.ols(formula='SO2~color_0+color_1+color_2+color_3+color_4+color_5+color_6+color_7+Fenghua_0+Type_0+Wenshi_1+Wenshi_2',data=alldata).fit()
model_SO2.summary()

test_data=dum_data[dum_data['Fenghua_0']==0]
test_data['Fenghua_0'].replace(0,1,inplace=True)

param=model_k2o.params.tolist() # sio2
b,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12=param
param=param[1:]

list_x=test_data.loc[1].tolist()

func=sum(np.multiply(param,list_x))+b

test_index=test_data.index.tolist()

def func(i):
    list_x=test_data.loc[i].tolist()
    res=sum(np.multiply(param,list_x))+b
    print(res)
for i in test_index:
    func(i)

