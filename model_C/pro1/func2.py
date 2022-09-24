from os import pread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
import torch.nn as nn
from scipy.optimize import curve_fit


#传入数据
data =pd.read_csv("/Users/steafen/Desktop/data2.csv")
#print(data.head())
x_data=data.iloc[:,-4:]
y_data=data.iloc[:,0]
#print(y_data)




# 可视化初步判断
sns.pairplot(data, x_vars=['Wenshi','Type','color','Fenghua'], y_vars='SiO2', size=7, aspect=0.8, kind='reg')  
#plt.show()  


# 进行多元函数拟合
# predictor=LinearRegression(n_jobs=-1)
# predictor.fit(X=x_data,y=y_data)
# coe=predictor.coef_
# print(coe)
# print(predictor.intercept_)
# w1,w2,w3,w4=coe
# b=predictor.intercept_


# res=np.linalg.lstsq(x_data,y_data,rcond=None)
# print(res[0])
# w1,w2,w3,w4=res[0]

def fun(x_data,coef_a,coef_b,coef_c,coef_d,intercept): #定义拟合的形式,第一个参数为自变量，后面是要拟合的参数
    a=x_data.iloc[:,0]
    b=x_data.iloc[:,1]
    c=x_data.iloc[:,2]
    d=x_data.iloc[:,3]
    return coef_a*np.exp(a)+coef_b*b+coef_c*c+coef_d*d+ intercept

res=curve_fit(fun,x_data,y_data,method="lm")
print(res[0])
w1,w2,w3,w4,b=res[0]


# 进行误差分析
loss=0
for i in range(len(x_data)):
    x1,x2,x3,x4=list(x_data[i:i+1].iloc[0])
    loss=loss+np.square(y_data[i]-(w1*np.exp(x1)+w2*x2+w3*x3+w4*x4+b))
print(loss)

stdloss=np.sqrt((loss/len(x_data)))
print(stdloss)








