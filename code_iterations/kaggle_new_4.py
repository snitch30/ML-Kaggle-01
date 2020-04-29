#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:56:02 2019

@author: stejasmunees
"""
test=0
#df['Married'].astype('category').cat.codes

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


print(df_label.isnull().sum())
print(df_label.isnull().values.any())
print(df_label.isnull().sum().sum())

#Removing all data points with more than 3 values missing at the same time
#We're doing this only for training data (~300)
df_label.dropna(thresh=9,inplace=True)

year_median=round(df_label['Year of Record'].median())
age_median=round(df_label['Age'].median())

################ REPEATING DATASET #################

#repeat=df_label.copy()
#repeat.dropna(inplace=True)
#
#print(repeat.isnull().sum())
#print(repeat.isnull().values.any())
#print(repeat.isnull().sum().sum())
#
#repeat_sample=repeat.sample(frac=0.3)
#repeat=repeat.drop(repeat_sample.index)
#test_sample=repeat.sample(frac=0.15)
#
#df_label = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv') 
#dup=df_label.copy()
#df_label.dropna(thresh=9,inplace=True)
#df_label=df_label.drop(test_sample.index)
#
#frame_ancient=[df_label,repeat_sample,repeat_sample]
#df_label = pd.concat(frame_ancient, sort=False,axis=0)
##We don't need instance value, hence we remove them.
#del df_label['Instance']

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
    df_label.dropna(subset=['Profession'],inplace=True)
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


################  FUNCTIONS FOR DATA ENCODING   ####################


#Label Encoding Profession Manually.   

def labelencode_proff(proff,proff_dict,counter):
    proff_le=[]
    for i in range(len(proff)):
        if proff[i] in proff_dict.keys():
            proff_le.append(proff_dict[proff[i]])
        if proff[i] not in proff_dict.keys():
            proff_dict[proff[i]]=counter
            counter=counter+1
            proff_le.append(proff_dict[proff[i]])
#    proff_le=np.array(proff_le)
#    proff_le=pd.DataFrame(proff_le)
#    proff_le.rename(
#            columns={
#                    0:"Profession"
#                    },inplace=True
#            )
    return proff_le,counter

#def onehot_cont(cont,dfc):
#    cont_ohe=pd.get_dummies(cont)
#    cont_ohe=cont_ohe.reindex(columns=dfc.columns,fill_value=0)
#    return cont_ohe
#
#def onehot_subreg(subreg,dfsr):
#    subreg_ohe=pd.get_dummies(subreg)
#    subreg_ohe=subreg_ohe.reindex(columns=dfsr.columns,fill_value=0)
#    return subreg_ohe
#
#def onehot_unideg(unid,dfu):
#    unideg_ohe=pd.get_dummies(unid)
#    unideg_ohe=unideg_ohe.reindex(columns=dfu.columns,fill_value=0)    
#    return unideg_ohe
#
#def onehot_hairval(hc,dfh):
#    hairval_ohe=pd.get_dummies(hc)
#    hairval_ohe=hairval_ohe.reindex(columns=dfh.columns,fill_value=0)    
#    return hairval_ohe
#
#def onehot_gender(g,dfg):
#    gender_ohe=pd.get_dummies(g)
#    gender_ohe=gender_ohe.reindex(columns=dfg.columns,fill_value=0)
#    return gender_ohe

############### DUMMY TESTING

def onehot_cont(cont,dfc):
    cont_ohe=pd.get_dummies(cont)
#    cont_ohe=cont_ohe.reindex(columns=dfc.columns,fill_value=0)
    return cont_ohe

def onehot_subreg(subreg,dfsr):
    subreg_ohe=pd.get_dummies(subreg)
#    subreg_ohe=subreg_ohe.reindex(columns=dfsr.columns,fill_value=0)
    return subreg_ohe

def onehot_unideg(unid,dfu):
    unideg_ohe=pd.get_dummies(unid)
#    unideg_ohe=unideg_ohe.reindex(columns=dfu.columns,fill_value=0)    
    return unideg_ohe

def onehot_hairval(hc,dfh):
    hairval_ohe=pd.get_dummies(hc)
#    hairval_ohe=hairval_ohe.reindex(columns=dfh.columns,fill_value=0)    
    return hairval_ohe

def onehot_gender(g,dfg):
    gender_ohe=pd.get_dummies(g)
#    gender_ohe=gender_ohe.reindex(columns=dfg.columns,fill_value=0)
    return gender_ohe


################ FUNCTION FOR FEATURE SCALING ############
    
def norma_feature(feature,mean,std):
    for i in range(len(feature)):
        feature[i]=(float(feature[i])-mean)/(std)
    return feature

#unique_vals=pd.Series(country).unique().tolist()
#unique_cont=cont_dat['sub_region'].unique().tolist()    

##############  Calling Functions to clean Data   #################
clean_gender(df_label)
clean_unideg(df_label)
clean_hairvals(df_label)
clean_year(df_label)
clean_age(df_label)
proff=clean_proff(df_label)


print(df_label.isnull().sum())
print(df_label.isnull().values.any())
print(df_label.isnull().sum().sum())

########## After Data Cleaning, we extract the data. ############

#Creating Dictionary with Indices for faster calculation
country_dict={}
for index in range(len(cont_dat['country'])):
    country_dict[cont_dat['country'][index]]=index

year=df_label.iloc[:,0].values
gender=df_label.iloc[:,1].values
age=df_label.iloc[:,2].values
country=df_label.iloc[:,3].values
sizeofcity=df_label.iloc[:,4].values
proff=pd.Series(proff)
unideg=df_label.iloc[:,6].values
glasses=df_label.iloc[:,7].values
hair_colour=df_label.iloc[:,8].values
body_height=df_label.iloc[:,9].values
y=df_label.iloc[:,10].values
continent,sub_region=sub_continent(country)

##################### OUTLIER REMOVAL ####################

#Removing 3 instances of salary greater than 3.0e+06
count=0
for i in range(len(y)):
    if (y[i]>2.75e06):
        y[i]=2.75e06
        count=count+1
##Removing 211 instances less than 4ft and 310 instances more than 7'6"ft
#less=0
#more=0
#for i in range(len(body_height)):
#    if (body_height[i]<122):
#        body_height[i]=122
#        less=less+1
#    if (body_height[i]>229):
#        body_height[i]=229
#        more=more+1
##Removing 45 instanes of ppl more than 100 years old
#less=0
#more=0
#for i in range(len(age)):
#    if (age[i]<15):
#        age[i]=1
#        less=less+1
#    if (age[i]>100):
#        age[i]=100
#        more=more+1
        
        
############## Calling Functions to encode Data ####################

#Creating Dictionary with Indices for label encoding
proff_dict={}
proff=np.array(proff)
counter_dict=1
for i in proff:
    if i not in proff_dict:
        proff_dict[i]=counter_dict
        counter_dict=counter_dict+1 
        
proff_le,counter_dict=labelencode_proff(proff,proff_dict,counter_dict)

#Creating Dummies Frame so that test and training data can be fit later

#Continent
unique_continents=pd.DataFrame(pd.Series(continent).unique().tolist())
unique_continents.rename(columns={0:"Continent"},inplace=True)
dummies_frame_continent = pd.get_dummies(unique_continents)
#Sub Continent
unique_sub_region=pd.DataFrame(pd.Series(sub_region).unique().tolist())
unique_sub_region.rename(columns={0:"Sub_Region"},inplace=True)
dummies_frame_sub_region = pd.get_dummies(unique_sub_region)
#University Degree
unique_degree=pd.DataFrame(pd.Series(unideg).unique().tolist())
unique_degree.rename(columns={0:"Uni_Deg"},inplace=True)
dummies_frame_unideg=pd.get_dummies(unique_degree)
#Hair Colour
unique_hair_colour=pd.DataFrame(pd.Series(hair_colour).unique().tolist())
unique_hair_colour.rename(columns={0:"Hair_Colour"},inplace=True)
dummies_frame_hairval=pd.get_dummies(unique_hair_colour)
#Gender
unique_gender=pd.DataFrame(pd.Series(gender).unique().tolist())
unique_gender.rename(columns={0:"Gender"},inplace=True)
dummies_frame_gender=pd.get_dummies(unique_gender)

#Encoding

#cont_ohe=onehot_cont(continent,dummies_frame_continent)
#del cont_ohe['Continent_Oceania']
#subreg_ohe=onehot_subreg(sub_region,dummies_frame_sub_region)
#del subreg_ohe['Sub_Region_Melanesia']
#unideg_ohe=onehot_unideg(unideg,dummies_frame_unideg)
#del unideg_ohe['Uni_Deg_Unknown']
#hairval_ohe=onehot_hairval(hair_colour,dummies_frame_hairval)
#del hairval_ohe['Hair_Colour_Unknown']
#gender_ohe=onehot_gender(gender,dummies_frame_gender)
#del gender_ohe['Gender_unknown']
#glass_ohe=pd.DataFrame(glasses)
#glass_ohe.rename(columns={0:"Glasses"},inplace=True)

# Dummy Testing
cont_ohe=onehot_cont(continent,dummies_frame_continent)
del cont_ohe['Oceania']
subreg_ohe=onehot_subreg(sub_region,dummies_frame_sub_region)
del subreg_ohe['Melanesia']
unideg_ohe=onehot_unideg(unideg,dummies_frame_unideg)
del unideg_ohe['Unknown']
hairval_ohe=onehot_hairval(hair_colour,dummies_frame_hairval)
del hairval_ohe['Unknown']
gender_ohe=onehot_gender(gender,dummies_frame_gender)
del gender_ohe['unknown']
glass_ohe=pd.DataFrame(glasses)
glass_ohe.rename(columns={0:"Glasses"},inplace=True)

#Feature Scaling (Standardization)
mean_year=np.mean(year)
std_year=np.std(year)
mean_age=np.mean(age)
std_age=np.std(age)
mean_csize=np.mean(sizeofcity)
std_csize=np.std(sizeofcity)
mean_bheight=np.mean(body_height)
std_bheight=np.std(body_height)
mean_proff=np.mean(proff_le)
std_proff=np.std(proff_le)

norm_year=list(norma_feature(year,mean_year,std_year))
norm_age=list(norma_feature(age,mean_age,std_age))
norm_csize=list(norma_feature(sizeofcity.astype(np.float64),mean_csize,std_csize))
norm_bheight=list(norma_feature(body_height.astype(np.float64),mean_bheight,std_bheight))
norm_proff_le=norma_feature(proff_le,mean_proff,std_proff)

norm_year_pd=pd.DataFrame(norm_year)
norm_year_pd.rename(columns={0:"Year"},inplace=True)
norm_age_pd=pd.DataFrame(norm_age)
norm_age_pd.rename(columns={0:"Age"},inplace=True)
norm_csize_pd=pd.DataFrame(norm_csize)
norm_csize_pd.rename(columns={0:"City_Size"},inplace=True)
norm_proff_le_pd=pd.DataFrame(norm_proff_le)
norm_proff_le_pd.rename(columns={0:"Profession"},inplace=True)

#Doing this to index y properly while merging the DataFrame to avoid nan
test=[]
for vals in df_label['Income in EUR']:
    test.append(vals)
y=pd.Series(test)

frames=[norm_year_pd,norm_age_pd,norm_csize_pd,norm_proff_le_pd,
        subreg_ohe,unideg_ohe,hairval_ohe,gender_ohe,glass_ohe]

result_new = pd.concat(frames, sort=False,axis=1)
result_backup=result_new.copy()

print(result_new.isnull().sum())
print(result_new.isnull().values.any())
print(result_new.isnull().sum().sum())
print(y.isnull().sum())
print(y.isnull().values.any())
print(y.isnull().sum().sum())

del result_new['Profession']
#################### TRAINING/FITTING #######################

from sklearn.model_selection import KFold
kf = KFold(n_splits = 10, random_state = 2,)
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import LinearSVR
# Linear Regression
regressor=LinearRegression()
linear_regression_scores=[]
# Stochaistic Gradient Descent
sgdr=SGDRegressor(alpha=0.001,max_iter=2000)
sgdr_scores=[]
# Support Vector Regressor
best_lsvr=LinearSVR(tol=1e-4,max_iter=2000, C=1)
linear_svr_scores=[]

#result = next(kf.split(result_new), None)
#print (result)
#train = result_new.iloc[result[0]]
#test =  result_new.iloc[result[1]]

counter=0
for train_index, test_index in kf.split(result_new):
    print("Train Index: ",train_index, "\n")
    print("Test Index: ", test_index)
    counter=counter+1
    print(counter)
    X_train,X_test,y_train,y_test = result_new.iloc[train_index], result_new.iloc[test_index],y[train_index],y[test_index]
    regressor.fit(X_train,y_train)
#    if(counter==4):
#        save_x=X_train.copy()
#        save_y=y_train.copy()
    sgdr.fit(X_train,y_train)
    best_lsvr.fit(X_train,y_train)
    linear_regression_scores.append(regressor.score(X_test,y_test))
    sgdr_scores.append(sgdr.score(X_test,y_test))
    linear_svr_scores.append(best_lsvr.score(X_test,y_test))
    
#################### XGBoost ###################
    
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

gbm = xgb.XGBRegressor()
reg_cv = GridSearchCV(gbm, {"colsample_bytree":[0.5,1.0],"min_child_weight":[1.0,1.2]
                            ,'max_depth': [3,4,6], 'n_estimators': [500,1000,2000]}, verbose=1)
reg_cv.fit(X_train,y_train)
reg_cv.best_params_

gmb = xgb.XGBRegressor(**reg_cv.best_params_)

model = xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=1000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 

#reg_cv.best_params_
#Out[270]: 
#{'colsample_bytree': 0.5,
# 'max_depth': 3,
# 'min_child_weight': 1.0,
# 'n_estimators': 1000}

model.fit(X_train,y_train) 

predictions = gbm.predict(X_test)
sc=model.score(X_test,y_test)

############ Linear Regression after Data Repeat ############

#X_test=test_sample.copy()
#pred_df=X_test.copy()   # Only to encode this test data, continued down
#
##### DO LINES BELOW IN PREDICITING TO ENCODE TEST DATA #####
#
#y_test=test_sample['Income in EUR']
#
##### Prediction Part #####
#
#from sklearn.model_selection import KFold
#from sklearn.linear_model import LinearRegression, SGDRegressor
#from sklearn.svm import LinearSVR
## Linear Regression
#regressor=LinearRegression()
#linear_regression_scores=[]
## Stochaistic Gradient Descent
#sgdr=SGDRegressor(alpha=0.001,max_iter=2000)
#sgdr_scores=[]
## Support Vector Regressor
#best_lsvr=LinearSVR(tol=1e-4,max_iter=2000, C=1)
#linear_svr_scores=[]
#
#regressor.fit(result_new,y)
#sgdr.fit(result_new,y)
#best_lsvr.fit(result_new,y)
#
#linear_regression_scores.append(regressor.score(result_pred,y_test))
#sgdr_scores.append(sgdr.score(result_pred,y_test))
#linear_svr_scores.append(best_lsvr.score(result_pred,y_test))

############################################################

#sgdr.fit(savex,savey)    
#regressor.fit(save_x,save_y)


##################### PREDICTING ######################

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

year_pred=pred_df.iloc[:,0].values
gender_pred=pred_df.iloc[:,1].values
age_pred=pred_df.iloc[:,2].values
country_pred=pred_df.iloc[:,3].values
sizeofcity_pred=pred_df.iloc[:,4].values
proff_pred=pd.Series(proff_pred)
unideg_pred=pred_df.iloc[:,6].values
glasses_pred=pred_df.iloc[:,7].values
hair_colour_pred=pred_df.iloc[:,8].values
body_height_pred=pred_df.iloc[:,9].values
y_pred=pred_df.iloc[:,10].values
continent_pred,sub_region_pred=sub_continent(country_pred)

proff_le_pred,counter_dict=labelencode_proff(proff_pred,proff_dict,counter_dict)

print(pd.Series(proff_le_pred).isnull().sum())
print(pd.Series(proff_le_pred).isnull().values.any())
print(pd.Series(proff_le_pred).isnull().sum().sum())

#Encoding

#cont_ohe_pred=onehot_cont(continent_pred,dummies_frame_continent)
#del cont_ohe_pred['Continent_Oceania']
#subreg_ohe_pred=onehot_subreg(sub_region_pred,dummies_frame_sub_region)
#del subreg_ohe_pred['Sub_Region_Melanesia']
#unideg_ohe_pred=onehot_unideg(unideg_pred,dummies_frame_unideg)
#del unideg_ohe_pred['Uni_Deg_Unknown']
#hairval_ohe_pred=onehot_hairval(hair_colour_pred,dummies_frame_hairval)
#del hairval_ohe_pred['Hair_Colour_Unknown']
#gender_ohe_pred=onehot_gender(gender_pred,dummies_frame_gender)
#del gender_ohe_pred['Gender_unknown']
#glass_ohe_pred=pd.DataFrame(glasses_pred)
#glass_ohe_pred.rename(columns={0:"Glasses"},inplace=True)

#Dummies Testing
cont_ohe_pred=onehot_cont(continent_pred,dummies_frame_continent)
del cont_ohe_pred['Oceania']
subreg_ohe_pred=onehot_subreg(sub_region_pred,dummies_frame_sub_region)
del subreg_ohe_pred['Melanesia']
unideg_ohe_pred=onehot_unideg(unideg_pred,dummies_frame_unideg)
del unideg_ohe_pred['Unknown']
hairval_ohe_pred=onehot_hairval(hair_colour_pred,dummies_frame_hairval)
del hairval_ohe_pred['Unknown']
gender_ohe_pred=onehot_gender(gender_pred,dummies_frame_gender)
del gender_ohe_pred['unknown']
glass_ohe_pred=pd.DataFrame(glasses_pred)
glass_ohe_pred.rename(columns={0:"Glasses"},inplace=True)


norm_year_pred=list(norma_feature(year_pred,mean_year,std_year))
norm_age_pred=list(norma_feature(age_pred,mean_age,std_age))
norm_csize_pred=list(norma_feature(sizeofcity_pred.astype(np.float64),mean_csize,std_csize))
norm_bheight_pred=list(norma_feature(body_height_pred.astype(np.float64),mean_bheight,std_bheight))
norm_proff_le_pred=norma_feature(proff_le_pred,mean_proff,std_proff)

norm_year_pd_pred=pd.DataFrame(norm_year_pred)
norm_year_pd_pred.rename(columns={0:"Year"},inplace=True)
norm_age_pd_pred=pd.DataFrame(norm_age_pred)
norm_age_pd_pred.rename(columns={0:"Age"},inplace=True)
norm_csize_pd_pred=pd.DataFrame(norm_csize_pred)
norm_csize_pd_pred.rename(columns={0:"City_Size"},inplace=True)
norm_proff_le_pd_pred=pd.DataFrame(norm_proff_le_pred)
norm_proff_le_pd_pred.rename(columns={0:"Profession"},inplace=True)

print(norm_proff_le_pd_pred.isnull().sum())
print(norm_proff_le_pd_pred.isnull().values.any())
print(norm_proff_le_pd_pred.isnull().sum().sum())

frame_pred=[norm_year_pd_pred,norm_age_pd_pred,norm_csize_pd_pred,
        norm_proff_le_pd_pred,subreg_ohe_pred,unideg_ohe_pred,
        hairval_ohe_pred,gender_ohe_pred,glass_ohe_pred]

len(norm_proff_le_pd_pred)

result_pred = pd.concat(frame_pred, sort=False,axis=1)

print(result_pred.isnull().sum())
print(result_pred.isnull().values.any())
print(result_pred.isnull().sum().sum())

#y_pred=regressor.predict(result_pred)
y_pred=gbm.predict(result_pred)
y_pred=model.predict(result_pred)
del result_pred['Profession']
y_pred=regressor.predict(result_pred)

sub=pd.read_csv('tcd ml 2019-20 income prediction submission file copy.csv')
del sub['Income']
inc=pd.DataFrame(y_pred)
framee=[sub,inc]
sub=pd.concat(framee,sort=False,axis=1)
sub.rename(
        columns={
                0:'Income'
                },inplace=True)


sub.to_csv('submission_file_08.csv',index=False)

import seaborn as sns
plt.figure(figsize=(40,40))
cor = result_new.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
plt.show()

#Plotting for checking Outliers
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(x = body_height, y=y)
plt.show()

#year=df_label.iloc[:,0].value_counts()
#gender=df_label.iloc[:,1].value_counts()
#age=df_label.iloc[:,2].value_counts()
#country=df_label.iloc[:,3].value_counts()
#sizeofcity=df_label.iloc[:,4].value_counts()
#unideg=df_label.iloc[:,6].value_counts()
#glasses=df_label.iloc[:,7].value_counts()
#hair_colour=df_label.iloc[:,8].value_counts()
#body_height=df_label.iloc[:,9].value_counts()
#y=df_label.iloc[:,10].value_counts()

#This is how you transform for encoding
#df1 = pd.get_dummies(pd.DataFrame({'cat':['a'],'val':[1]}))
#df1.reindex(columns = dummies_frame.columns, fill_value=0)

#    yes=0
#    no=0
#    cont_not=[]
#    for cname in unique_vals:
#        if (cname in list(cont_dat['country'])):
#            yes=yes+1
#        else:
#            no=no+1
#            cont_not.append(cname)
