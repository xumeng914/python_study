# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:30:30 2017

@author: xumeng
"""

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
 
# 加载数据
# 备用地址: http://cdn.powerxing.com/files/lr-binary.csv
df = pd.read_csv("http://cdn.powerxing.com/files/lr-binary.csv")
#df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")
 
# 浏览数据集
df.head()
# 重命名'rank'列，因为dataframe中有个方法名也为'rank'
df.columns = ["admit", "gre", "gpa", "prestige"]
df.columns
# array([admit, gre, gpa, prestige], dtype=object)

# summarize the data
df.describe()

# 查看每一列的标准差
df.std()

# 频率表，表示prestige与admin的值相应的数量关系，类似R中的table函数
pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])

df.hist()
pl.show()

# 将prestige设为虚拟变量
dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
dummy_ranks.head()
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
data.head()

# 需要自行添加逻辑回归所需的intercept变量
data['intercept'] = 1.0

# 指定作为训练变量的列，不含目标列`admit`
train_cols = data.columns[1:]
# Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)
 
logit = sm.Logit(data['admit'], data[train_cols])
 
# 拟合模型
result = logit.fit()

# 查看数据的要点
result.summary()


# 构建预测集
# 与训练集相似，一般也是通过 pd.read_csv() 读入
# 在这边为方便，我们将训练集拷贝一份作为预测集（不包括 admin 列）
import copy
combos = copy.deepcopy(data)
 
# 数据中的列要跟预测时用到的列一致
predict_cols = combos.columns[1:]
 
# 预测集也要添加intercept变量
combos['intercept'] = 1.0
 
# 进行预测，并将预测评分存入 predict 列中
combos['predict'] = result.predict(combos[predict_cols])

# 预测完成后，predict 的值是介于 [0, 1] 间的概率值
# 我们可以根据需要，提取预测结果
# 例如，假定 predict > 0.5，则表示会被录取
# 在这边我们检验一下上述选取结果的精确度
total = 0
hit = 0
for value in combos.values:
  # 预测分数 predict, 是数据中的最后一列
  predict = value[-1]
  # 实际录取结果
  admit = int(value[0])
 
  # 假定预测概率大于0.5则表示预测被录取
  if predict > 0.5:
    total += 1
    # 表示预测命中
    if admit == 1:
      hit += 1
 
# 输出结果
print ('Total: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0*hit/total) )
# Total: 49, Hit: 30, Precision: 61.22





