#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:11:50 2019

@author: stejasmunees
"""

from scipy.stats.mstats import winsorize
winsorize(pd.Series(range(20), dtype='float'), limits=[0.05, 0.05])


test=winsorize(df['Body Height [cm]'], limits=[0.05, 0.05])
plt.scatter(y,test)

test=0
testagain=0


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,20))
cor = result.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

y=range(len(df['Body Height [cm]']))

df = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv') 

dup=df
#df= df.dropna(axis='rows')
'''
This has 
    print(df.isnull().sum())
Instance                0
Year of Record        441
Gender               7432
Age                   494
Country                 0
Size of City            0
Profession            322
University Degree    7370
Wears Glasses           0
Hair Color           7242
Body Height [cm]        0
Income in EUR           0
dtype: int64
We can try replacing mean, median or mode for year, age

median = df['Year of Record'].median()
df['Year of Record'].fillna(median, inplace=True)
'''

#yor_index=[]
#count=0
#for index in df['Year of Record'].isnull():
#    if index==True:
#        yor_index.append(count)
#    count=count+1
    
year_median=round(df['Year of Record'].median())
df['Year of Record'].fillna(year_median,inplace=True)
df['Gender'].fillna("unknown",inplace=True)
df['Gender'].replace('0',"unknown",inplace=True)
df['Hair Color'].fillna("Unknown",inplace=True)
df['Hair Color'].replace('0',"Unknown",inplace=True)
age_median=round(df['Age'].median())
df['Age'].fillna(age_median,inplace=True)
#df['Profession'].fillna(method='bfill',inplace=True)
#df['Profession'].fillna(method='ffill',inplace=True)
df['Profession'].fillna("Unknown",inplace=True)
df['University Degree'].fillna("Unknown",inplace=True)
df['University Degree'].replace('0',"Unknown",inplace=True)
print(df.isnull().sum())

#df= df[df.Gender != '0']
##df= df[df.Gender != 'unknown']
#df= df[df['Hair Color'] != '0']
##df= df[df['Hair Color'] != 'Unknown']
#df= df[df['University Degree'] != '0']

#This has to be done in the test data as well.
instance_vals=df['Instance'].value_counts()
year_vals=df['Year of Record'].value_counts()
gender_vals=df['Gender'].value_counts()
age_vals=df['Age'].value_counts()
country_vals=df['Country'].value_counts()
citysize_vals=df['Size of City'].value_counts()
proff_vals=df['Profession'].value_counts()
unideg_vals=df['University Degree'].value_counts()
glass_vals=df['Wears Glasses'].value_counts()
hair_vals=df['Hair Color'].value_counts()
height_vals=df['Body Height [cm]'].value_counts()

instances=df.iloc[:,0].values
year=df.iloc[:,1].values
gender=df.iloc[:,2].values
age=df.iloc[:,3].values
country=df.iloc[:,4].values
sizeofcity=df.iloc[:,5].values
proff=df.iloc[:,6].values
unideg=df.iloc[:,7].values
glasses=df.iloc[:,8].values
hair_colour=df.iloc[:,9].values
body_height=df.iloc[:,10].values
y=df.iloc[:,11].values

del df['Instance']


for i in range(len(proff)):
    proff[i]=str.lower(proff[i])
    
proff_dict={}

counter=1
for i in proff:
    if i not in proff_dict:
        proff_dict[i]=counter
        counter=counter+1
       
proff_le=[]
for i in range(len(proff)):
    if proff[i] in proff_dict.keys():
        proff_le.append(proff_dict[proff[i]])
        if proff[i] not in proff_dict.keys():
            proff_dict[proff[i]]=counter
            counter=counter+1
            proff_le.append(proff_dict[proff[i]])
            
proff_le=np.array(proff_le)
proff_le=pd.DataFrame(proff_le)
proff_le.rename(
        columns={
                0:"Profession"
                },inplace=True
        )
#year_vals
#gender_vals
#age_vals
#country_vals
#citysize_vals
#proff_vals
#unideg_vals
#glass_vals
#hair_vals
#height_vals

print(df.isnull().sum())
print(df.isnull().values.any())
print(df.isnull().sum().sum())
dup_enc=df.copy()


from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from sklearn.compose import ColumnTransformer,make_column_transformer

#gender_le = LabelEncoder()
#gender_le_=gender_le.fit_transform(dup_enc['Gender'])
#dup_enc['Gender']=gender_le_

#stackoverflow.com/questions/28465633/easy-way-to-apply-transformation-from-pandas-get-dummies-to-new-data
#gender_ohe_=pd.get_dummies(gender)
#gender_ohe_=gender_ohe_.drop(columns='unknown')
#
#df1=pd.get_dummies(pd.DataFrame(Input))
#gender_frame=gender_ohe_

#Gender OneHotEncoding
gender_ohe=OneHotEncoder()
gender_ohe_ = gender_ohe.fit_transform(gender.reshape(-1,1))
gender_ohe_ = pd.DataFrame(gender_ohe_.toarray())
gender_ohe_.columns = gender_ohe.get_feature_names()
#Remove Unknown Gender
gender_ohe_ = gender_ohe_.drop(columns=gender_ohe.get_feature_names()[3])
#University Degree OneHotEncoding
unideg_ohe=OneHotEncoder()
unideg_ohe_ = unideg_ohe.fit_transform(unideg.reshape(-1,1))
unideg_ohe_ = pd.DataFrame(unideg_ohe_.toarray())
unideg_ohe_.columns = unideg_ohe.get_feature_names()
#Remove Unknown Degree
unideg_ohe_ = unideg_ohe_.drop(columns=unideg_ohe.get_feature_names()[4])
#Work Label Encoding
#proff_le=LabelEncoder()
#proff_le_=pd.DataFrame(proff_le.fit_transform(proff))
#proff_le_.rename(columns={0: "Profession"},inplace=True)
#Country OneHotEncoding
country_ohe=OneHotEncoder(handle_unknown='ignore')
country_ohe_ = country_ohe.fit_transform(country.reshape(-1,1))
country_ohe_ = pd.DataFrame(country_ohe_.toarray())
country_ohe_.columns = country_ohe.get_feature_names()
#Remove One Country
country_ohe_ = country_ohe_.drop(columns=country_ohe.get_feature_names()[0])
#Temporary Country Label Encoding
#country_le=LabelEncoder()
#country_le_=pd.DataFrame(country_le.fit_transform(country))
#country_le_.rename(columns={0: "Country"},inplace=True)
#Hair Colour OneHotEncoding
haircolour_ohe=OneHotEncoder()
haircolour_ohe_ = haircolour_ohe.fit_transform(hair_colour.reshape(-1,1))
haircolour_ohe_ = pd.DataFrame(haircolour_ohe_.toarray())
haircolour_ohe_.columns = haircolour_ohe.get_feature_names()
#Remove Unknown Degree
haircolour_ohe_ = haircolour_ohe_.drop(columns=haircolour_ohe.get_feature_names()[4])

#Removing the above's equivalent old data from the complete dataframe's copy.
dup_enc = dup_enc.drop(columns=["Gender","Country","Profession","University Degree","Hair Color"])

#Merging all
frames = [dup_enc, gender_ohe_, unideg_ohe_,pd.DataFrame(proff_le),country_ohe_,haircolour_ohe_]
result = pd.concat(frames, sort=False,axis=1)
y_result=result["Income in EUR"]
result=result.drop("Income in EUR",axis=1)

#Mean Normalising
#result["Year of Record"]=(result["Year of Record"]-result["Year of Record"].mean())/result["Year of Record"].std()

year_min=result["Year of Record"].min()
year_max=result["Year of Record"].max()
age_min=result["Age"].min()
age_max=result["Age"].max()
proff_min=result["Profession"].min()
proff_max=result["Profession"].max()
citysize_min=result["Size of City"].min()
citysize_max=result["Size of City"].max()
bodyheight_min=result["Body Height [cm]"].min()
bodyheight_max=result["Body Height [cm]"].max()

#Min Max Normalising
result["Year of Record"]=(result["Year of Record"]-year_min)/(year_max-year_min)
result["Age"]=(result["Age"]-age_min)/(age_max-age_min)
result["Profession"]=(result["Profession"]-proff_min)/(proff_max-proff_min)
result["Size of City"]=(result["Size of City"]-citysize_min)/(citysize_max-citysize_min)
result["Body Height [cm]"]=(result["Body Height [cm]"]-bodyheight_min)/(bodyheight_max-bodyheight_min)
#result["Country"]=(result["Country"]-result["Country"].min())/(result["Country"].max()-result["Country"].min())


#country_le = LabelEncoder()
#country_le_ = country_le.fit_transform(country)
#dup_enc['Country']=country_le_

#################### TRAINING/FITTING #######################

#K-Fold Cross Validation
from sklearn.model_selection import KFold
#from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
import numpy as np

result_array=result.as_matrix()
linear_regression_scores=[]
linear_svr_scores=[]
best_lsvr=LinearSVR(tol=1e-5)
regressor=LinearRegression()
cv=KFold(n_splits=25)
counter=0
for train_index, test_index in cv.split(result):
    print("Train Index: ",train_index, "\n")
    print("Test Index: ", test_index)
    counter=counter+1
    print(counter)
    X_train, X_test, y_train, y_test = result_array[train_index], result_array[test_index], y[train_index],y[test_index]
    regressor.fit(X_train,y_train)
    best_lsvr.fit(X_train,y_train)
    if(counter==6):
        save_model=regressor
        save_X_train, save_X_test, save_y_train, save_y_test=X_train, X_test, y_train, y_test
    linear_regression_scores.append(regressor.score(X_test,y_test))
    linear_svr_scores.append(best_lsvr.score(X_test,y_test))

regressor.fit(save_X_test,save_y_test)

##### PREDICTING #####

df = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv') 
year_median=round(df['Year of Record'].median())
df['Year of Record'].fillna(year_median,inplace=True)
df['Gender'].fillna("unknown",inplace=True)
df['Gender'].replace('0',"unknown",inplace=True)
df['Hair Color'].fillna("Unknown",inplace=True)
df['Hair Color'].replace('0',"Unknown",inplace=True)
age_median=round(df['Age'].median())
df['Age'].fillna(age_median,inplace=True)
#df['Profession'].fillna(method='bfill',inplace=True)
#df['Profession'].fillna(method='ffill',inplace=True)
df['Profession'].fillna("Unknown",inplace=True)
df['University Degree'].fillna("Unknown",inplace=True)
df['University Degree'].replace('0',"Unknown",inplace=True)
print(df.isnull().sum())

instance_vals=df['Instance'].value_counts()
year_vals=df['Year of Record'].value_counts()
gender_vals=df['Gender'].value_counts()
age_vals=df['Age'].value_counts()
country_vals=df['Country'].value_counts()
citysize_vals=df['Size of City'].value_counts()
proff_vals=df['Profession'].value_counts()
unideg_vals=df['University Degree'].value_counts()
glass_vals=df['Wears Glasses'].value_counts()
hair_vals=df['Hair Color'].value_counts()
height_vals=df['Body Height [cm]'].value_counts()

instances=df.iloc[:,0].values
year=df.iloc[:,1].values
gender=df.iloc[:,2].values
age=df.iloc[:,3].values
country=df.iloc[:,4].values
sizeofcity=df.iloc[:,5].values
proff=df.iloc[:,6].values
unideg=df.iloc[:,7].values
glasses=df.iloc[:,8].values
hair_colour=df.iloc[:,9].values
body_height=df.iloc[:,10].values
del df['Instance']

print(df.isnull().sum())
print(df.isnull().values.any())
print(df.isnull().sum().sum())
dup_enc=df.copy()

#Gender OneHotEncoding
gender_ohe_ = gender_ohe.transform(gender.reshape(-1,1))
gender_ohe_ = pd.DataFrame(gender_ohe_.toarray())
gender_ohe_.columns = gender_ohe.get_feature_names()
#Remove Unknown Gender
gender_ohe_ = gender_ohe_.drop(columns=gender_ohe.get_feature_names()[3])
#University Degree OneHotEncoding
unideg_ohe_ = unideg_ohe.transform(unideg.reshape(-1,1))
unideg_ohe_ = pd.DataFrame(unideg_ohe_.toarray())
unideg_ohe_.columns = unideg_ohe.get_feature_names()
#Remove Unknown Degree
unideg_ohe_ = unideg_ohe_.drop(columns=unideg_ohe.get_feature_names()[4])


for i in range(len(proff)):
    proff[i]=str.lower(proff[i])

proff_le=[]    

for i in range(len(proff)):
    if proff[i] in proff_dict.keys():
        proff_le.append(proff_dict[proff[i]])
        if proff[i] not in proff_dict.keys():
            proff[i]=0


proff_le=np.array(proff_le)
proff_le=pd.DataFrame(proff_le)
proff_le.rename(
        columns={
                0:"Profession"
                },inplace=True
        )

#Work Label Encoding #We're using manual encoding using dictionaries
#proff_le_=pd.DataFrame(proff_le.transform(proff))
#proff_le_.rename(columns={0: "Profession"},inplace=True)

#Country OneHotEncoding
#country_ohe_ = country_ohe.transform(country.reshape(-1,1))
#country_ohe_ = pd.DataFrame(country_ohe_.toarray())
#country_ohe_.columns = country_ohe.get_feature_names()
#Remove One Country
#country_ohe_ = country_ohe_.drop(columns=country_ohe.get_feature_names()[0])

country_le_=pd.DataFrame(country_le.transform(country))
country_le_.rename(columns={0: "Country"},inplace=True)

#Hair Colour OneHotEncoding
haircolour_ohe_ = haircolour_ohe.transform(hair_colour.reshape(-1,1))
haircolour_ohe_ = pd.DataFrame(haircolour_ohe_.toarray())
haircolour_ohe_.columns = haircolour_ohe.get_feature_names()
#Remove Unknown Degree
haircolour_ohe_ = haircolour_ohe_.drop(columns=haircolour_ohe.get_feature_names()[4])

#Removing the above's equivalent old data from the complete dataframe's copy.
dup_enc = dup_enc.drop(columns=["Gender","Country","Profession","University Degree","Hair Color"])

#Merging all
frames = [dup_enc, gender_ohe_, unideg_ohe_,proff_le,country_le_,haircolour_ohe_]

new_result = pd.concat(frames, sort=False,axis=1)
del new_result['Income']
new_result.fillna(0,inplace=True)


new_result["Year of Record"]=(new_result["Year of Record"]-year_min)/(year_max-year_min)
new_result["Age"]=(new_result["Age"]-age_min)/(age_max-age_min)
new_result["Profession"]=(new_result["Profession"]-proff_min)/(proff_max-proff_min)
new_result["Size of City"]=(new_result["Size of City"]-citysize_min)/(citysize_max-citysize_min)
new_result["Body Height [cm]"]=(new_result["Body Height [cm]"]-bodyheight_min)/(bodyheight_max-bodyheight_min)



print(new_result.isnull().sum())
print(new_result.isnull().values.any())
print(new_result.isnull().sum().sum())

y=regressor.predict(new_result)


sub=pd.read_csv('tcd ml 2019-20 income prediction submission file copy.csv')
del sub['Income']
inc=pd.DataFrame(y)
framee=[sub,inc]
sub=pd.concat(framee,sort=False,axis=1)
sub.rename(
        columns={
                0:'Income'
                },inplace=True)


sub.to_csv('submission_file_01.csv',index=False)

#a=['male']

#testing single
#temp_var=['male']
#pd.DataFrame(gender_ohe.transform(np.array(temp_var).reshape(-1,1)).toarray())

#X = pd.get_dummies(dup_enc, prefix_sep='_', drop_first=True)

#sku=df.iloc[:,0].values
#sku=list(sku)
#base_colour=df.iloc[:,1].values
#base_colour=list(base_colour)
#craft=df.iloc[:,3].values
#craft=list(craft)
#zari=df.iloc[:,4].values
#zari=list(zari)
#body_pattern=df.iloc[:,5].values
#body_pattern=list(body_pattern)
#pallu_pattern=df.iloc[:,6].values
#pallu_pattern=list(pallu_pattern)

#Gender OneHotEncoding
#preprocessor_gender = make_column_transformer( (OneHotEncoder(categories='auto'),[1]),remainder="passthrough")
#dup_enc = preprocessor_gender.fit_transform(dup_enc)
#dup_enc = pd.DataFrame(dup_enc)

#
#dup_enc.rename(
#    columns={
#        0: "Gender_Female",
#        1: "Gender_Male",
#        2: "Gender_Other",
#        3: "Gender_Unknown",
#        4: "Year",
#        5: "Age",
#        6: "Country",
#        7: "Size_of_City",
#        8: "Profession",
#        9: "University_Degree",
#        10: "Wears_Glasses",
#        11: "Hair_Color",
#        12: "Body_Height"
#    },
#    inplace=True
#)
