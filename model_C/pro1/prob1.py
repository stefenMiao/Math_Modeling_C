from turtle import title
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
df =pd.read_csv("/Users/steafen/Desktop/data1.csv",header=0,index_col=0,) # 导入数据

plt.subplot(221)

count_orn=df['B'].value_counts().sort_index().tolist() # 绘图（纹饰）
label_orn=['A','B','C']
bar1=plt.bar(range(len(count_orn)),count_orn,tick_label=label_orn)
plt.bar_label(bar1,label_type='edge')
# plt.show()

plt.subplot(222)
count_type=df['C'].value_counts().sort_index().tolist() # 绘图（类型）
label_type=['高钾','铅钡']
bar2=plt.bar(range(len(count_type)),count_type,tick_label=label_type)
plt.bar_label(bar2,label_type='edge')
#plt.show()

plt.subplot(223)
count_color=df['D'].value_counts().sort_index().tolist() # 绘图（类型）
label_color=['空','蓝绿','浅蓝','深蓝','浅绿','深绿','紫','黑','绿']
bar3=plt.bar(range(len(count_color)),count_color,tick_label=label_color)
plt.bar_label(bar3,label_type='edge')
#plt.show()

plt.subplot(224)
count_fenghua=df['E'].value_counts().sort_index().tolist() # 绘图（类型）
label_fenghua=['未风化','已风化']
bar4=plt.bar(range(len(count_fenghua)),count_fenghua,tick_label=label_fenghua)
plt.bar_label(bar4,label_type='edge')
plt.show()


