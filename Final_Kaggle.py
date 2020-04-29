# -*- coding: utf-8 -*-
"""ML_Kaggle.ipynb

[link text](https://)# New Section
"""

# from catboost import CatBoostRegressor
!pip install catboost


# import os
# cd='/content/drive/My Drive/ml_colab'
# os.listdir(cd)

countrycsv=cd+'/country_cont copy.csv'
inccsv=cd+'/tcd ml 2019-20 income prediction training (with labels).csv'
predcsv=cd+'/tcd ml 2019-20 income prediction test (without labels).csv'
subcsv=cd+'/tcd ml 2019-20 income prediction submission file copy.csv'

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

################   LOAD DATA   ####################
df_label = pd.read_csv(inccsv) 
dup=df_label.copy()
##We don't need instance value, hence we remove them. 
#Remove this line if repeating Dataset
del df_label['Instance']
cont_dat= pd.read_csv(countrycsv)

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

############## ENCODING ##################

# year= df_label['Year of Record']
# age= df_label['Age']
# soc= df_label['Size of City']

# num_frame=[year,age,soc]
# num_data=pd.concat(num_frame, sort=False,axis=1)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaled_num_data=scaler.fit_transform(num_data)

# gender= df_label['Gender']
# country=df_label['Country']
# proff=df_label['Profession']
# uni_deg=df_label['University Degree']

# #3,7,166,1485 columns to removed after onehotencoding
# cat_frame=[gender,uni_deg,country,proff]
# cat_data=pd.concat(cat_frame,sort=False,axis=1)


# #from sklearn.preprocessing import StandardScaler
# #scaler = StandardScaler()
# #scaled_num_data=scaler.fit_transform(num_data)

# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder(handle_unknown='ignore')
# scaled_cat=enc.fit_transform(cat_data)
# scaled_cat_df=pd.DataFrame(scaled_cat.toarray())

# del scaled_cat_df[3]
# del scaled_cat_df[8]
# del scaled_cat_df[168]
# del scaled_cat_df[1488]

# final_frame=[pd.DataFrame(scaled_num_data),scaled_cat_df]
# final_dat=pd.concat(final_frame,sort=False,axis=1)

# #



# #test=[]
# #for vals in df_label['Income in EUR']:
# #    test.append(vals)
# #y=pd.Series(test)

# print(final_dat.isnull().sum())
# print(final_dat.isnull().values.any())
# print(final_dat.isnull().sum().sum())


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

y_temp=np.array(y_new)
high_count=0
low_count=0
y_try=[]
flag1=0
flag2=0

for i in range(len(y_temp)):
  flag1=0
  flag2=0
  if(y_temp[i]>=2.75e+06):
    high_count=high_count+1
    y_try.append(2.75e+06)
    flag1=1
  if(y_temp[i]<0):
    low_count=low_count+1
    y_try.append(0)
    flag2=1
  if(flag1==0 and flag2==0):
    y_try.append(y_temp[i])

for i in range(len(y_temp)):
  if(y_try[i]>=2.75e+06):
    high_count=high_count+1
#     y_temp[i]=2.75e+06
  if(y_try[i]<0):
    low_count=low_count+1
#     y_temp[i]=0

y_new=pd.Series(y_try)

len(df_label)

soc=df_label['Size of City']
year= df_label['Year of Record']
age= df_label['Age']
bh=df_label['Body Height [cm]']

del df_label['Profession']
del df_label['Size of City']
del df_label['Year of Record']
del df_label['Age']
del df_label['Body Height [cm]']

len(df_label)

indices=[]
for i in range(len(soc)):
    indices.append(i)
indices=pd.Index(indices)

indices

num_frame=[year,age,soc,bh]
num_data=pd.concat(num_frame, sort=False,axis=1)
num_data=num_data.set_index(indices)
proff=pd.DataFrame(proff).set_index(indices)
proff.rename(columns={0:"Profession"},inplace=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_num_data=scaler.fit_transform(num_data)
scaled_num_data=pd.DataFrame(scaled_num_data)
scaled_num_data=scaled_num_data.set_index(indices)

scaled_num_data

df_label=df_label.set_index(indices)

frame_some=[scaled_num_data,df_label,proff]
df_label=pd.concat(frame_some,sort=False,axis=1)

df_label

print(df_label.isnull().sum())
print(df_label.isnull().values.any())
print(df_label.isnull().sum().sum())

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

df_label.dtypes

from sklearn.model_selection import train_test_split
X_train_1,X_test_1,y_train_1,y_test_1=train_test_split(df_label,y_new,test_size=0.2,random_state=123)
X_train_2,X_test_2,y_train_2,y_test_2=train_test_split(only_cont,y_new,test_size=0.2,random_state=123)
X_train_3,X_test_3,y_train_3,y_test_3=train_test_split(only_sub,y_new,test_size=0.2,random_state=123)
X_train_4,X_test_4,y_train_4,y_test_4=train_test_split(all_dat,y_new,test_size=0.2,random_state=123)

categorical_features_indices1 = np.where(df_label.dtypes != np.float)[0]
categorical_features_indices2 = np.where(only_cont.dtypes != np.float)[0]
categorical_features_indices3 = np.where(only_sub.dtypes != np.float)[0]
categorical_features_indices4 = np.where(all_dat.dtypes != np.float)[0]

from catboost import CatBoostRegressor

model1=CatBoostRegressor(iterations=5000, depth=8, learning_rate=0.1, loss_function='RMSE',od_type='IncToDec',od_pval=0.001)
model2=CatBoostRegressor(iterations=1000, depth=16, learning_rate=0.1, loss_function='RMSE',od_type='IncToDec',od_pval=0.001)
model3=CatBoostRegressor(iterations=1000, depth=16, learning_rate=0.1, loss_function='RMSE',od_type='IncToDec',od_pval=0.001)
model4=CatBoostRegressor(iterations=1000, depth=16, learning_rate=0.1, loss_function='RMSE',od_type='IncToDec',od_pval=0.001)

model1.fit(X_train_1, y_train_1,cat_features=categorical_features_indices1,eval_set=(X_train_1, y_train_1),plot=True)
# model2.fit(X_train_2, y_train_2,cat_features=categorical_features_indices2,eval_set=(X_train_2, y_train_2),plot=True)
# model3.fit(X_train_3, y_train_3,cat_features=categorical_features_indices3,eval_set=(X_train_3, y_train_3),plot=True)
# model4.fit(X_train_4, y_train_4,cat_features=categorical_features_indices4,eval_set=(X_train_4, y_train_4),plot=True)

import pickle
pickle.dump(model1,open( "model5.pickle", "wb" ))

################  PREDICTIONS #########################
model1=pickle.load( open( "model5.pickle", "rb" ) )

pred_df = pd.read_csv(predcsv) 

del pred_df['Instance']

print(pred_df.isnull().sum())
print(pred_df.isnull().values.any())
print(pred_df.isnull().sum().sum())

clean_gender(pred_df)
clean_unideg(pred_df)
clean_hairvals(pred_df)
clean_year(pred_df)
clean_age(pred_df)
pred_df['Profession'].fillna("Unknown",inplace=True)
proff_pred=clean_proff(pred_df)

print(pred_df.isnull().sum())
print(pred_df.isnull().values.any())
print(pred_df.isnull().sum().sum())

pred_df = pred_df.astype({"Size of City": float, "Body Height [cm]": float})

pred_df['Income']

soc_pred=pred_df['Size of City']
year_pred= pred_df['Year of Record']
age_pred= pred_df['Age']
bh_pred=pred_df['Body Height [cm]']

del pred_df['Profession']
del pred_df['Size of City']
del pred_df['Year of Record']
del pred_df['Age']
del pred_df['Body Height [cm]']

indices_pred=[]
for i in range(len(soc_pred)):
    indices_pred.append(i)
indices_pred=pd.Index(indices_pred)

num_frame_pred=[year_pred,age_pred,soc_pred,bh_pred]
num_data_pred=pd.concat(num_frame_pred, sort=False,axis=1)
num_data_pred=num_data_pred.set_index(indices_pred)
proff_pred=pd.DataFrame(proff_pred).set_index(indices_pred)
proff_pred.rename(columns={0:"Profession"},inplace=True)

scaled_num_data_pred=scaler.transform(num_data_pred)
scaled_num_data_pred=pd.DataFrame(scaled_num_data_pred)
scaled_num_data_pred=scaled_num_data_pred.set_index(indices_pred)

pred_df=pred_df.set_index(indices_pred)

frame_some_pred=[scaled_num_data_pred,pred_df,proff_pred]
pred_df=pd.concat(frame_some_pred,sort=False,axis=1)

print(pred_df.isnull().sum())
print(pred_df.isnull().values.any())
print(pred_df.isnull().sum().sum())

del pred_df['Income']
pred_df.dtypes

y_pred=model1.predict(pred_df)

sub=pd.read_csv(subcsv)
del sub['Income']
inc=pd.DataFrame(y_pred)
framee=[sub,inc]
sub=pd.concat(framee,sort=False,axis=1)
sub.rename(
        columns={
                0:'Income'
                },inplace=True)


sub.to_csv('submission_file_12.csv',index=False)
