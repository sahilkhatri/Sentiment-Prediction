# -*- coding: utf-8 -*-
"""
Created on Sat May 23 18:37:40 2020

@author: Sahil
"""

import pandas as pd
import os

os.chdir('E:\\silvertouch\\aclImdb')
print(len(os.listdir()))

"""
    for generating train dataset 
"""
os.chdir('E:\\silvertouch\\aclImdb\\train\\pos')
print(len(os.listdir()))
train_pos_list = []
for i in os.listdir():
    f = open(i,"r", encoding="utf8")
    train_pos_list.append(f.readline())
    f.close()
train_data_pos = pd.DataFrame(columns = ['review','sentiment'])
train_data_pos['review']=train_pos_list
train_data_pos['sentiment']=1

os.chdir('E:\\silvertouch\\aclImdb\\train\\neg')
print(len(os.listdir()))
train_neg_list = []
for i in os.listdir():
    f = open(i,"r", encoding="utf8")
    train_neg_list.append(f.readline())
    f.close()
train_data_neg = pd.DataFrame(columns = ['review','sentiment'])
train_data_neg['review']=train_neg_list
train_data_neg['sentiment']=0

train_data = pd.concat([train_data_pos, train_data_neg], axis=0)
train_data = train_data.sample(frac=1).reset_index(drop=True)

os.chdir('E:\\silvertouch\\aclImdb')
train_data.to_csv('train.csv', index=False)


"""
    for generating test dataset
"""
os.chdir('E:\\silvertouch\\aclImdb\\test\\pos')
print(len(os.listdir()))
test_pos_list = []
for i in os.listdir():
    f = open(i,"r", encoding="utf8")
    test_pos_list.append(f.readline())
    f.close()
test_data_pos = pd.DataFrame(columns = ['review','sentiment'])
test_data_pos['review']=test_pos_list
test_data_pos['sentiment']=1

os.chdir('E:\\silvertouch\\aclImdb\\test\\neg')
print(len(os.listdir()))
test_neg_list = []
for i in os.listdir():
    f = open(i,"r", encoding="utf8")
    test_neg_list.append(f.readline())
    f.close()
test_data_neg = pd.DataFrame(columns = ['review','sentiment'])
test_data_neg['review']=test_neg_list
test_data_neg['sentiment']=0

test_data = pd.concat([test_data_pos, test_data_neg], axis=0)
test_data = test_data.sample(frac=1).reset_index(drop=True)

os.chdir('E:\\silvertouch\\aclImdb')
test_data.to_csv('test.csv', index=False)


