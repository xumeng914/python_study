# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:21:55 2017

@author: xumeng
"""

import pandas as pd
import numpy as np
import os
os.chdir('d:/python/mt_xykfq')
data=pd.read_csv('d:/python/mt_xykfq/data/LoanStats2016Q4.csv',skiprows=1)

data.head()
data.shape

import missingno
missingno.matrix(data)

check_null=data.isnull().sum(axis=0).sort_values(ascending=False)/float(len(data))
check_null[check_null > 0]

data=data.dropna(thresh=len(data)*0.5, axis = 1)
data=data.loc[:,data.apply(pd.Series.nunique)!=1]

data.describe()

data['loan_amnt'].describe()
data.loc[0:2]
data.index

from sklearn.linear_model import LogisticRegression
















from numpy import * 
from sklearn.datasets import load_iris     # import datasets

# load the dataset: iris
iris = load_iris() 
samples = iris.data
#print samples 
target = iris.target 

# import the LogisticRegression
from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression()  # 使用类，参数全是默认的
classifier.fit(samples, target)  # 训练数据来学习，不需要返回值

x = classifier.predict([5, 3, 5, 2.5])  # 测试数据，分类返回标记

print(x) 

#其实导入的是sklearn.linear_model的一个类：LogisticRegression， 它里面有许多方法
#常用的方法是fit（训练分类模型）、predict（预测测试样本的标记）

#不过里面没有返回LR模型中学习到的权重向量w，感觉这是一个缺陷











