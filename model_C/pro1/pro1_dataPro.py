from turtle import title
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
df =pd.read_csv("/Users/steafen/Desktop/data1.csv",header=0,index_col=0,) # 导入数据

df_fenghua=df[df['E']==1] # 筛选出已经风化的数据
print(df_fenghua)

plt.subplot(311)
plt.title('已风化数据')
count_orn=df_fenghua['B'].value_counts().sort_index().tolist() # 绘图（纹饰）
label_orn=['A','B','C']
bar1=plt.bar(range(len(count_orn)),count_orn,tick_label=label_orn)
plt.bar_label(bar1,label_type='edge')
# plt.show()

plt.subplot(312)
count_type=df_fenghua['C'].value_counts().sort_index().tolist() # 绘图（类型）
label_type=['高钾','铅钡']
bar2=plt.bar(range(len(count_type)),count_type,tick_label=label_type)
plt.bar_label(bar2,label_type='edge')
#plt.show()

plt.subplot(313)
count_color=df_fenghua['D'].value_counts().sort_index()#.tolist() # 绘图（类型）
print(count_color)
label_color=['空','蓝绿','浅蓝','深蓝','浅绿','深绿','紫','黑','绿']
count_color=[4,9,12,0,1,4,2,2,0]
bar3=plt.bar(range(len(count_color)),count_color,tick_label=label_color)
plt.bar_label(bar3,label_type='edge')
plt.show()