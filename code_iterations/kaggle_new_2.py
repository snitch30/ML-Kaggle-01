#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:25:32 2019

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

############# FUNCTIONS FOR CREATING NEW FEATURES ####################

def sub_continent(cont):
    conti=[]
    sub_conti=[]
    
    for countries in cont:
        sub_cont_index=country_dict[countries]
        conti.append(cont_dat['continent'][sub_cont_index])
        sub_conti.append(cont_dat['sub_region'][sub_cont_index])
    return conti, sub_conti

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


############ Dropping Unneccary Columns ############
#df_label=df.copy()
del df_label['Wears Glasses']
del df_label['Hair Color']
del df_label['Body Height [cm]']

############## ENCODING ##################

year= df_label['Year of Record']
age= df_label['Age']
soc= df_label['Size of City']

num_frame=[year,age,soc]
num_data=pd.concat(num_frame, sort=False,axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_num_data=scaler.fit_transform(num_data)

gender= df_label['Gender']
country=df_label['Country']
proff=df_label['Profession']
uni_deg=df_label['University Degree']

#3,7,166,1485 columns to removed after onehotencoding
cat_frame=[gender,uni_deg,country,proff]
cat_data=pd.concat(cat_frame,sort=False,axis=1)


#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaled_num_data=scaler.fit_transform(num_data)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
scaled_cat=enc.fit_transform(cat_data)
scaled_cat_df=pd.DataFrame(scaled_cat.toarray())

del scaled_cat_df[3]
del scaled_cat_df[8]
del scaled_cat_df[168]
del scaled_cat_df[1488]

final_frame=[pd.DataFrame(scaled_num_data),scaled_cat_df]
final_dat=pd.concat(final_frame,sort=False,axis=1)

#



#test=[]
#for vals in df_label['Income in EUR']:
#    test.append(vals)
#y=pd.Series(test)

print(final_dat.isnull().sum())
print(final_dat.isnull().values.any())
print(final_dat.isnull().sum().sum())

#################### TRAINING/FITTING #######################

from sklearn.model_selection import KFold
kf = KFold(n_splits = 10, random_state = 2,)
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import LinearSVR

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(final_dat,y_new[0],test_size=0.2,random_state=0)

# Linear Regression
regressor=LinearRegression()
linear_regression_scores=[]
# Stochaistic Gradient Descent
sgdr=SGDRegressor(alpha=0.001,max_iter=2000)
sgdr_scores=[]
# Support Vector Regressor
best_lsvr=LinearSVR(tol=1e-4,max_iter=2000, C=1)
linear_svr_scores=[]
#Random Forest
from sklearn.ensemble import RandomForestRegressor
for_reg=RandomForestRegressor(n_estimators=10,random_state=0)
forest_scores=[]

#result = next(kf.split(result_new), None)
#print (result)
#train = result_new.iloc[result[0]]
#test =  result_new.iloc[result[1]]

counter=0
for train_index, test_index in kf.split(final_dat):
    print("Train Index: ",train_index, "\n")
    print("Test Index: ", test_index)
    counter=counter+1
    print(counter)
    X_train,X_test,y_train,y_test = final_dat.iloc[train_index], final_dat.iloc[test_index],y_new[0][train_index],y_new[0][test_index]
#    regressor.fit(X_train,y_train)
#    if(counter==4):
#        save_x=X_train.copy()
#        save_y=y_train.copy()
#    sgdr.fit(X_train,y_train)
#    best_lsvr.fit(X_train,y_train)
    for_reg.fit(X_train,y_train)
    forest_scores.append(for_reg.score(X_test,y_test))
    print(forest_scores[counter])
    
#    linear_regression_scores.append(regressor.score(X_test,y_test))
#    sgdr_scores.append(sgdr.score(X_test,y_test))
#    linear_svr_scores.append(best_lsvr.score(X_test,y_test))



################  PREDICTIONS #########################

pred_df = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv') 

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

############ Dropping Unneccary Columns ############
#df_label=df.copy()
del pred_df['Wears Glasses']
del pred_df['Hair Color']
del pred_df['Body Height [cm]']

############## ENCODING ##################

year_pred= pred_df['Year of Record']
age_pred= pred_df['Age']
soc_pred= pred_df['Size of City']

num_frame_pred=[year_pred,age_pred,soc_pred]
num_data_pred=pd.concat(num_frame_pred, sort=False,axis=1)

gender_pred= pred_df['Gender']
country_pred=pred_df['Country']
proff_pred=pred_df['Profession']
uni_deg_pred=pred_df['University Degree']

#3,7,166,1485 columns to removed after onehotencoding
cat_frame_pred=[gender_pred,uni_deg_pred,country_pred,proff_pred]
cat_data_pred=pd.concat(cat_frame_pred,sort=False,axis=1)

#from sklearn.preprocessing import StandardScaler
scaled_num_data_pred=scaler.transform(num_data_pred)

scaled_cat_pred=enc.transform(cat_data_pred)
scaled_cat_df_pred=pd.DataFrame(scaled_cat_pred.toarray())

del scaled_cat_df_pred[3]
del scaled_cat_df_pred[8]
del scaled_cat_df_pred[168]
del scaled_cat_df_pred[1488]

final_frame_pred=[pd.DataFrame(num_data_pred),scaled_cat_df_pred]
final_dat_pred=pd.concat(final_frame_pred,sort=False,axis=1)

print(final_dat_pred.isnull().sum())
print(final_dat_pred.isnull().values.any())
print(final_dat_pred.isnull().sum().sum())

###################################################

y_pred=for_reg.predict(final_dat_pred)

sub=pd.read_csv('tcd ml 2019-20 income prediction submission file copy.csv')
del sub['Income']
inc=pd.DataFrame(y_pred)
framee=[sub,inc]
sub=pd.concat(framee,sort=False,axis=1)
sub.rename(
        columns={
                0:'Income'
                },inplace=True)


sub.to_csv('submission_file_07.csv',index=False)


print(X_train.isnull().sum())
print(X_train.isnull().values.any())
print(X_train.isnull().sum().sum())

print(y_train.isnull().sum())
print(y_train.isnull().values.any())
print(y_train.isnull().sum().sum())