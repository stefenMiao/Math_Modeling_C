#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
import random
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.decomposition import PCA
from operator import mul

data =pd.read_csv("/Users/steafen/Desktop/data2.csv")
data.drop(['Wenshi','color'],axis=1,inplace=True)

data_K=data[data['Type']==0] # 筛选高钾类型
data_K0=data_K[data_K['Fenghua']==0] # 筛选出高钾未分化
data_K1=data_K[data_K['Fenghua']==1] # 筛选出高钾已分化
data_Pb=data[data['Type']==1]
data_Pb0=data_Pb[data_Pb['Fenghua']==0]
data_Pb1=data_Pb[data_Pb['Fenghua']==1]

# 数据归一化
trainK1_scaled=normalize(data_K1)
trainK1_scaled=pd.DataFrame(trainK1_scaled,columns=data_K1.columns)
traindata_K1=trainK1_scaled.drop(['Type','Fenghua'],axis=1)

trainK0_scaled=normalize(data_K0)
trainK0_scaled=pd.DataFrame(trainK0_scaled,columns=data_K0.columns)
traindata_K0=trainK0_scaled.drop(['Type','Fenghua'],axis=1)

trainPb1_scaled=normalize(data_Pb1)
trainPb1_scaled=pd.DataFrame(trainPb1_scaled,columns=data_Pb1.columns)
traindata_Pb1=trainPb1_scaled.drop(['Type','Fenghua'],axis=1)

trainPb0_scaled=normalize(data_Pb0)
trainPb0_scaled=pd.DataFrame(trainPb0_scaled,columns=data_Pb0.columns)
traindata_Pb0=trainPb0_scaled.drop(['Type','Fenghua'],axis=1)


# PCA 高钾未风化

pca_K0=PCA(n_components=3)
pca_K0.fit(traindata_K0)
print(pca_K0.explained_variance_ratio_)
print(pca_K0.explained_variance_)
print(pca_K0.n_components)
print(pca_K0.components_)  # 因子载荷矩阵/ 成分矩阵


K0_spss=pca_K0.components_/np.sqrt(pca_K0.explained_variance_.reshape(pca_K0.n_components_,1))
K0_spss  # 成分得分系数矩阵---各自因子载荷向量除以各自因子特征值的算数平方根  --代表系数


for i in range(3):
    print(K0_spss[i][[1,2,3,5,6,7]])

# 最后主成分得分=标准化后的因子得分系数×解释方差比例
K0_score=pca_K0.transform(traindata_K0)    
print('因子得分：',K0_score)
# 因子得分归一化，使其满足正态分布
K0_scaler2=StandardScaler().fit(K0_score)
K0_scaler2=pd.DataFrame(K0_scaler2.transform(K0_score),columns=['FAC1','FAC2','FAC3'])
# 正负号转换
K0_sign=np.sign(K0_spss.sum(axis=1))
# 取正负号
K0_scaler2_sign=K0_scaler2*K0_sign
# 综合得分
K0_rate=pca_K0.explained_variance_ratio_
K0_scaler2_sign['FAC_score']=np.sum(K0_scaler2_sign*K0_rate,axis=1)
print('主成分得分矩阵K0')
print(K0_scaler2_sign) # 主成分得分矩阵


print(traindata_K0)


# PCA 高钾已风化

pca_K1=PCA(n_components=2)
pca_K1.fit(traindata_K1)
print(pca_K1.components_)  
print(pca_K1.explained_variance_ratio_)
K1_spss=pca_K1.components_/np.sqrt(pca_K1.explained_variance_.reshape(pca_K1.n_components_,1))
K1_spss # 成分得分矩阵---各自因子载荷向量除以各自因子特征值的算数平方根  --判断主成分代表的变量

for i in range(2):
    print(K1_spss[i][[2,3,4,5,7]])

# 最后主成分得分=标准化后的因子得分系数×解释方差比例
K1_score=pca_K1.transform(traindata_K1)    
print('因子得分：',K1_score)
# 因子得分归一化，使其满足正态分布
K1_scaler2=StandardScaler().fit(K1_score)
K1_scaler2=pd.DataFrame(K1_scaler2.transform(K1_score),columns=['FAC1','FAC2'])
# 正负号转换
K1_sign=np.sign(K1_spss.sum(axis=1))
# 取正负号
K1_scaler2_sign=K1_scaler2*K1_sign
# 综合得分
K1_rate=pca_K1.explained_variance_ratio_
K1_scaler2_sign['FAC_score']=np.sum(K1_scaler2_sign*K1_rate,axis=1)
print('主成分得分矩阵K1')
print(K1_scaler2_sign) # 主成分得分矩阵当做系数
print(K1_scaler2)
print(traindata_K1)

# 铅钡未风化

pca_Pb0=PCA(n_components=2)
pca_Pb0.fit(traindata_Pb0)
print(pca_Pb0.components_)
print(pca_Pb0.explained_variance_ratio_)
Pb0_spss=pca_Pb0.components_/np.sqrt(pca_Pb0.explained_variance_.reshape(pca_Pb0.n_components_,1))
Pb0_spss # 成分得分矩阵---各自因子载荷向量除以各自因子特征值的算数平方根  --判断主成分代表的变量

# 最后主成分得分=标准化后的因子得分系数×解释方差比例
Pb0_score=pca_Pb0.transform(traindata_Pb0)    
print('因子得分：',Pb0_score)
# 因子得分归一化，使其满足正态分布
Pb0_scaler2=StandardScaler().fit(Pb0_score)
Pb0_scaler2=pd.DataFrame(Pb0_scaler2.transform(Pb0_score),columns=['FAC1','FAC2'])
# 正负号转换
Pb0_sign=np.sign(Pb0_spss.sum(axis=1))
# 取正负号
Pb0_scaler2_sign=Pb0_scaler2*Pb0_sign
# 综合得分
Pb0_rate=pca_Pb0.explained_variance_ratio_
Pb0_scaler2_sign['FAC_score']=np.sum(Pb0_scaler2_sign*Pb0_rate,axis=1)
print('主成分得分矩阵Pb0')
print(Pb0_scaler2_sign) # 主成分得分矩阵当做系数
for i in range(2):
    print(Pb0_spss[i][[0,7,8,9]])

# 铅钡已风化
pca_Pb1=PCA(n_components=2)
pca_Pb1.fit(traindata_Pb1)
print(pca_Pb1.components_)
print(pca_Pb1.explained_variance_ratio_)
Pb1_spss=pca_Pb1.components_/np.sqrt(pca_Pb1.explained_variance_.reshape(pca_Pb1.n_components_,1))
Pb1_spss # 成分得分矩阵---各自因子载荷向量除以各自因子特征值的算数平方根  --判断主成分代表的变量

for i in range(2):
    print(Pb1_spss[i][[0,8,9,13]])

# 最后主成分得分=标准化后的因子得分系数×解释方差比例
Pb1_score=pca_Pb1.transform(traindata_Pb1)    
print('因子得分：',Pb1_score)
# 因子得分归一化，使其满足正态分布
Pb1_scaler2=StandardScaler().fit(Pb1_score)
Pb1_scaler2=pd.DataFrame(Pb1_scaler2.transform(Pb1_score),columns=['FAC1','FAC2'])
# 正负号转换
Pb1_sign=np.sign(Pb1_spss.sum(axis=1))
# 取正负号
Pb1_scaler2_sign=Pb1_scaler2*Pb1_sign
# 综合得分
Pb1_rate=pca_Pb1.explained_variance_ratio_
Pb1_scaler2_sign['FAC_score']=np.sum(Pb1_scaler2_sign*Pb1_rate,axis=1)
print('主成分得分矩阵Pb1')
print(Pb1_scaler2_sign) # 主成分得分矩阵当做系数

# 因子得分公式

# 高钾未风化
Pb0_w=Pb0_scaler2_sign.loc[:,'FAC1'].tolist()

def yinzi_Pb0(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12):
    Pb0_x=[0,x2,x3,x4,0,x6,x7,x8,0,0,0,0]
    return mul(Pb0_w,Pb0_x)

