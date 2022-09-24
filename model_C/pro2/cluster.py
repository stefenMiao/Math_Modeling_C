#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
import random
import scipy.cluster.hierarchy as shc

data =pd.read_csv("/Users/steafen/Desktop/data2.csv")
data.drop(['Wenshi','color'],axis=1,inplace=True)

data_K=data[data['Type']==0] # 筛选高钾类型
data_K0=data_K[data_K['Fenghua']==0] # 筛选出高钾未分化
data_K1=data_K[data_K['Fenghua']==1] # 筛选出高钾已分化

data_Pb=data[data['Type']==1]
data_Pb0=data_Pb[data_Pb['Fenghua']==0]
data_Pb1=data_Pb[data_Pb['Fenghua']==1]
data_Pb0

K0_std=data_K0.std() # 求标准差
K1_std=data_K1.std()
Pb0_std=data_Pb0.std()
Pb1_std=data_Pb1.std()

df1=K0_std.to_frame()
col1=df1.values
index1=df1.index.tolist()


df2=K1_std.to_frame()
col2=df2.values
index2=df2.index.tolist()

df3=Pb0_std.to_frame()
col3=df3.values
index3=df3.index.tolist()


df4=Pb1_std.to_frame()
col4=df4.values
index4=df4.index.tolist()

plt.figure(figsize=(20,20))
plt.subplot(221)  # 绘制标准差散点图
plt.scatter(x=index1,y=col1) # 高钾未风化
plt.subplot(222)
plt.scatter(x=index2,y=col2) # 高钾风化
plt.subplot(223)
plt.scatter(x=index3,y=col3) # 铅钡未风化
plt.subplot(224)
plt.scatter(x=index4,y=col4) # 铅钡风化


# 高钾未风化取**sio2** \\
# 高钾已风化取**sio2 Al2O3 CuO** \\
# 铅钡未风化取**SiO2 PbO BaO** \\
# 铅钡已风化取**SiO2 PbO** \\

train_K1=data_K1.loc[:,['SiO2','Al2O3','CuO']]

# 进行数据归一化 确保数据的尺度相同
from sklearn.preprocessing import normalize
trainK1_scaled=normalize(train_K1)
trainK1_scaled=pd.DataFrame(trainK1_scaled,columns=train_K1.columns)


# 绘制树状图分析簇的数量
plt.figure(figsize=(10,7))
plt.title('K1_Dendrograms')

# 决定阈值为0.03 切断树状图
plt.axhline(y=0.025,color='r',linestyle='--')
dend_K1=shc.dendrogram(shc.linkage(trainK1_scaled,method='ward'))


# 使用层次聚类进行分析
from sklearn.cluster import AgglomerativeClustering
K1_cluster=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
K1_cluster.fit_predict(trainK1_scaled)

plt.figure(figsize=(10,7))
plt.scatter(trainK1_scaled['SiO2'],trainK1_scaled['Al2O3'],c=K1_cluster.labels_)

from mpl_toolkits.mplot3d import Axes3D

fig_K1 = plt.figure(figsize=(10,7))
ax_K1 = Axes3D(fig_K1)
ax_K1.scatter(trainK1_scaled['SiO2'],trainK1_scaled['Al2O3'],trainK1_scaled['CuO'],c=K1_cluster.labels_)

# 对铅钡已风化进行层次聚类

train_Pb1=data_Pb1.loc[:,['SiO2','PbO']]

# 进行数据归一化 确保数据的尺度相同
trainPb1_scaled=normalize(train_Pb1)
trainPb1_scaled=pd.DataFrame(trainPb1_scaled,columns=train_Pb1.columns)

# 绘制树状图分析簇的数量
plt.figure(figsize=(10,7))
plt.title('Pb1_Dendrograms')
# 决定阈值为2 切断树状图
plt.axhline(y=2,color='r',linestyle='--')
dend_K1=shc.dendrogram(shc.linkage(trainPb1_scaled,method='ward'))

# 使用层次聚类进行分析
Pb1_cluster=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
Pb1_cluster.fit_predict(trainPb1_scaled)

plt.figure(figsize=(10,7))
plt.scatter(trainPb1_scaled['SiO2'],trainPb1_scaled['PbO'],c=Pb1_cluster.labels_)

# 对铅钡未风化做聚类分析
train_Pb0=data_Pb0.loc[:,['SiO2','PbO','BaO']]


# 进行数据归一化 确保数据的尺度相同
trainPb0_scaled=normalize(train_Pb0)
trainPb0_scaled=pd.DataFrame(trainPb0_scaled,columns=train_Pb0.columns)


# 绘制树状图分析簇的数量
plt.figure(figsize=(10,7))
plt.title('Pb0_Dendrograms')
# 决定阈值为2 切断树状图
plt.axhline(y=0.8,color='r',linestyle='--')
dend_K1=shc.dendrogram(shc.linkage(trainPb0_scaled,method='ward'))

# 使用层次聚类进行分析
Pb0_cluster=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
Pb0_cluster.fit_predict(trainPb0_scaled)

fig_Pb0 = plt.figure(figsize=(10,7))
ax_Pb0 = Axes3D(fig_Pb0)
ax_Pb0.scatter(trainPb0_scaled['SiO2'],trainPb0_scaled['PbO'],trainPb0_scaled['BaO'],c=Pb0_cluster.labels_)

train_K0=data_K0.loc[:,'SiO2']#.values
# 轮廓系数评估
metrics.silhouette_score(train_K1,K1_cluster.labels_,metric='euclidean') # 高钾已风化
metrics.silhouette_score(train_Pb0,Pb0_cluster.labels_,metric='euclidean') # 铅钡未风化
metrics.silhouette_score(train_Pb1,Pb1_cluster.labels_,metric='euclidean') # 铅钡已风化

# kmeans聚类

# 高钾已风化
km_K1=KMeans(n_clusters=3)
km_K1.fit(trainK1_scaled)
km_K1.cluster_centers_

km_K1.labels_
# 铅钡未风化
km_Pb0=KMeans(n_clusters=2)
km_Pb0.fit(trainPb0_scaled)
km_Pb0.cluster_centers_

# 铅钡已风化
km_Pb1=KMeans(n_clusters=2)
km_Pb1.fit(trainPb1_scaled)
km_Pb1.cluster_centers_


print(metrics.silhouette_score(train_K1,km_K1.labels_,metric='euclidean'))
print(metrics.silhouette_score(train_Pb0,km_Pb0.labels_,metric='euclidean'))
print(metrics.silhouette_score(train_Pb1,km_Pb1.labels_,metric='euclidean'))

K1_sio2=train_K1.loc[:,'SiO2'].tolist()
train_K1.index


mean_K1_sio2=np.mean(K1_sio2)
K1_sio2_noise=[]
K1_sio2_noise



for i  in range(len(K1_sio2)):
    K1_sio2_noise.append(K1_sio2[i]+random.uniform(-mean_K1_sio2*0.05,mean_K1_sio2*0.05))
K1_sio2_noise     # 获取带噪声的数据 sio2               
    


K1_al=train_K1.loc[:,'Al2O3'].tolist()
K1_cuo=train_K1.loc[:,'CuO'].tolist()

mean_K1_al=np.mean(K1_al)
K1_al_noise=[]

mean_K1_cuo=np.mean(K1_cuo)
K1_cuo_noise=[]


for i  in range(len(K1_al)):
    K1_al_noise.append(K1_al[i]+random.uniform(-mean_K1_al*0.05,mean_K1_al*0.05))
print(K1_al_noise)  # 噪音al2o3

for i  in range(len(K1_cuo)):
    K1_cuo_noise.append(K1_cuo[i]+random.uniform(-mean_K1_cuo*0.05,mean_K1_cuo*0.05))
print(K1_cuo_noise )  # 噪音cuo

col=['SiO2','Al2O3','CuO']
K1_dict={'SiO2':K1_sio2_noise,'Al2O3':K1_al_noise,'CuO':K1_cuo_noise}
K1_noise=pd.DataFrame(K1_dict)
K1_noise

rate_K1=sum(km_K1.predict(K1_noise)==km_K1.predict(trainK1_scaled))/len(K1_noise)

Pb1_si=train_Pb1.loc[:,'SiO2'].tolist()
Pb1_pbo=train_Pb1.loc[:,'PbO'].tolist()

mean_Pb1_si=np.mean(Pb1_si)
Pb1_si_noise=[]

mean_Pb1_pbo=np.mean(Pb1_pbo)
Pb1_pbo_noise=[]
len(Pb1_si)

for i  in range(36):
    Pb1_si_noise.append(Pb1_si[i]+random.uniform(-mean_Pb1_si*0.1,mean_Pb1_si*0.1))
print(len(Pb1_si_noise))  # 噪音sio2

for i  in range(36):
    Pb1_pbo_noise.append(Pb1_pbo[i]+random.uniform(-mean_Pb1_pbo*0.1,mean_Pb1_pbo*0.1))
print(len(Pb1_pbo_noise )  )# 噪音pbo

col=['SiO2','PbO']
Pb1_dict={'SiO2':Pb1_si_noise,'PbO':Pb1_pbo_noise}
Pb1_noise=pd.DataFrame(Pb1_dict)
rate_Pb1=sum(km_Pb1.predict(Pb1_noise)==km_Pb1.predict(trainPb1_scaled))/len(Pb1_noise)
# 对表三进行亚类划分
data=pd.read_csv("/Users/steafen/Desktop/data4.csv")

# 取高钾已风化
data_K=data[data['Type']==0]
data_K1=data_K[data_K['Fenghua']==1]
test_K1=data_K1.loc[:,['SiO2','Al2O3','CuO']]
km_K1.predict(test_K1)

# 取铅钡未风化
data_Pb=data[data['Type']==1]
data_Pb0=data_Pb[data_Pb['Fenghua']==0]
data_Pb0
test_Pb0=data_Pb0.loc[:,['SiO2','PbO','BaO']]

km_Pb0.predict(test_Pb0)

# 取铅钡已风化
data_Pb1=data_Pb[data_Pb['Fenghua']==1]
test_Pb1=data_Pb1.loc[:,['SiO2','PbO']]
km_Pb1.predict(test_Pb1)
