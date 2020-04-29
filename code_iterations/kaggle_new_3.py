#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:55:59 2019

@author: stejasmunees
"""

test=0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

################   LOAD DATA   ####################
df_label = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv') 
dup=df_label.copy()
##We don't need instance value, hence we remove them. 
#Remove this line if repeating Dataset
del df_label['Instance']
cont_dat= pd.read_csv('country_cont copy.csv')

country_dict={}
for index in range(len(cont_dat['country'])):
    country_dict[cont_dat['country'][index]]=index

print(df_label.isnull().sum())
print(df_label.isnull().values.any())
print(df_label.isnull().sum().sum())

#Removing all data points with more than 3 values missing at the same time
#We're doing this only for training data (~300)
df_label.dropna(thresh=9,inplace=True)

year_median=round(df_label['Year of Record'].median())
age_median=round(df_label['Age'].median())


################ FUNCTIONS FOR DATA CLEANING ####################

#Filling 441 nans and 724 '0' to unknown 
def clean_gender (df):
    df['Gender'].fillna("unknown",inplace=True)
    df['Gender'].replace('0',"unknown",inplace=True)
    return None

#Filling 7370 nans and 697 '0' to unknown
def clean_unideg (df):
    df['University Degree'].fillna("Unknown",inplace=True)
    df['University Degree'].replace('0',"Unknown",inplace=True)
    return None

#Wears Glasses, Body Height, City Size is clean in both training and test data.
#However have to check the correlation for feature extraction
    
#Filling 7242 nans and 29 '0' to unknown (655 unknown exist already)
def clean_hairvals (df):
    df['Hair Color'].fillna("Unknown",inplace=True)
    df['Hair Color'].replace('0',"Unknown",inplace=True)

#Filling 441 nans (295 in testing) with median    
def clean_year (df):
    df['Year of Record'].fillna(year_median,inplace=True)

#Filling 494 nans (279 in testing) with median
def clean_age (df):
    df['Age'].fillna(age_median,inplace=True)
    
#Filling 322 nans (195 in testing) with bfill
def clean_proff (df):
    df.dropna(subset=['Profession'],inplace=True)
#    df_label['Profession'].dropna(axis='index',inplace=True)
    proff=df.iloc[:,5].values
    for i in range(len(proff)):
        proff[i]=str.lower(proff[i])
        if (proff[i][-1:]==' '):
            proff[i]=proff[i][:-1]
        if (proff[i]=='.net software developer'):
            proff[i]=='.net developer'
        if (proff[i]=='account manager'):
            proff[i]=='accounts manager'
        if (proff[i]=='sewer'):
            proff[i]=='sewer pipe cleaner'
        if (proff[i][:8]=='accounts'):
            proff[i]=proff[i][:7]+proff[i][8:]
        if (proff[i][:3]=='sr.'):
            proff[i]='senior'+proff[i][3:]
        if (proff[i][:14]=='administration'):
            proff[i]='administrative'+proff[i][14:]
        if (proff[i][:14]=='staff analyst '):
            proff[i]='staff analyst 2'
    return proff

#len(df_label['Profession'].unique().tolist())
#len(pd.Series(proff).unique().tolist())
    
print(df_label.isnull().sum())
print(df_label.isnull().values.any())
print(df_label.isnull().sum().sum())

#######################################################

##############  Calling Functions to clean Data   #################
clean_gender(df_label)
clean_unideg(df_label)
clean_hairvals(df_label)
clean_year(df_label)
clean_age(df_label)
proff=clean_proff(df_label)

y_new=np.array(df_label['Income in EUR'])
y_new=pd.DataFrame(y_new)
print(y_new.isnull().sum())
print(y_new.isnull().values.any())
print(y_new.isnull().sum().sum())

print(df_label.isnull().sum())
print(df_label.isnull().values.any())
print(df_label.isnull().sum().sum())

############# FUNCTIONS FOR CREATING NEW FEATURES ####################

def sub_continent(cont):
    conti=[]
    sub_conti=[]
    
    for countries in cont:
        sub_cont_index=country_dict[countries]
        conti.append(cont_dat['continent'][sub_cont_index])
        sub_conti.append(cont_dat['sub_region'][sub_cont_index])
    return conti, sub_conti

###################################################################

continent,sub_region=sub_continent(df_label['Country'])
continent=pd.DataFrame(continent)
sub_region=pd.DataFrame(sub_region)
continent.rename(columns={0:"Continent"},inplace=True)
sub_region.rename(columns={0:"Sub_Region"},inplace=True)



df_label = df_label.astype({"Size of City": float, "Body Height [cm]": float})
no_country_label=df_label.copy()
del no_country_label['Country']

y_new=df_label['Income in EUR']
del df_label['Income in EUR']

############### RESETTING INDICES TO AVOID ISSUES WHILE MERGING 
indices=[]
for i in range(len(continent)):
    indices.append(i)
indices=pd.Index(indices)
df_label=df_label.set_index(indices)
continent=continent.set_index(indices)
sub_region=sub_region.set_index(indices)
no_country_label=no_country_label.set_index(indices)

only_cont_frame=[no_country_label,continent]
only_sub_frame=[no_country_label,sub_region]
all_frame=[df_label,continent,sub_region]

only_cont=pd.concat(only_cont_frame,sort=False,axis=1)
only_sub=pd.concat(only_sub_frame,sort=False,axis=1)
all_dat=pd.concat(all_frame,sort=False,axis=1)

only_cont=only_cont.astype({"Wears Glasses": int})
only_sub=only_sub.astype({"Wears Glasses": int})
all_dat=all_dat.astype({"Wears Glasses": int})

print(df_label.isnull().sum())
print(df_label.isnull().values.any())
print(df_label.isnull().sum().sum())

print(only_cont.isnull().sum())
print(only_cont.isnull().values.any())
print(only_cont.isnull().sum().sum())
    
print(only_sub.isnull().sum())
print(only_sub.isnull().values.any())
print(only_sub.isnull().sum().sum())

print(all_dat.isnull().sum())
print(all_dat.isnull().values.any())
print(all_dat.isnull().sum().sum())

from sklearn.model_selection import train_test_split
X_train_1,X_test_1,y_train_1,y_test_1=train_test_split(df_label,y_new,test_size=0.2,random_state=123)
X_train_2,X_test_2,y_train_2,y_test_2=train_test_split(only_cont,y_new,test_size=0.2)
X_train_3,X_test_3,y_train_3,y_test_3=train_test_split(only_sub,y_new,test_size=0.2)
X_train_4,X_test_4,y_train_4,y_test_4=train_test_split(all_dat,y_new,test_size=0.2)

categorical_features_indices1 = np.where(df_label.dtypes != np.float)[0]
categorical_features_indices2 = np.where(only_cont.dtypes != np.float)[0]
categorical_features_indices3 = np.where(only_sub.dtypes != np.float)[0]
categorical_features_indices4 = np.where(all_dat.dtypes != np.float)[0]

from catboost import CatBoostRegressor

model1=CatBoostRegressor(iterations=1000, depth=16, learning_rate=0.1, loss_function='RMSE')
model2=CatBoostRegressor(iterations=1000, depth=16, learning_rate=0.1, loss_function='RMSE')
model3=CatBoostRegressor(iterations=1000, depth=16, learning_rate=0.1, loss_function='RMSE')
model4=CatBoostRegressor(iterations=1000, depth=16, learning_rate=0.1, loss_function='RMSE')

model1.fit(X_train_1, y_train_1,cat_features=categorical_features_indices1,eval_set=(X_train_1, y_train_1),plot=True)
model2.fit(X_train_2, y_train_2,cat_features=categorical_features_indices2,eval_set=(X_train_2, y_train_2),plot=True)
model3.fit(X_train_3, y_train_3,cat_features=categorical_features_indices3,eval_set=(X_train_3, y_train_3),plot=True)
model4.fit(X_train_4, y_train_4,cat_features=categorical_features_indices4,eval_set=(X_train_4, y_train_4),plot=True)