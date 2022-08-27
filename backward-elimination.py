import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv('veriler.csv')

country = datas.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(datas.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()


c = datas.iloc[:,-1:].values


le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(datas.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
age = datas.iloc[:,1:4].values
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])

result = pd.DataFrame(data=country, index= range(22), columns = ['fr','tr','us'])

result2 = pd.DataFrame(data=age, index= range(22), columns = ['boy','kilo','yas'])

sex = datas.iloc[:,-1].values

result3 = pd.DataFrame(data = c[:,:1], index= range(22), columns = ['cinsiyet'])

s = pd.concat([result, result2], axis=1)


s2 = pd.concat([s,result3], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
'''
x_train, x_test, y_train, y_test = train_test_split(s,result3,test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
'''


len = s2.iloc[:,3:4].values

left = s2.iloc[:,:3]
right = s2.iloc[:,4:]

datas2 = pd.concat([left,right], axis=1)
print(datas2)

x_train, x_test, y_train, y_test = train_test_split(datas2,len,test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)

import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values = datas2, axis = 1)
X_l = datas2.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(len, X_l).fit()
print(model.summary())

X_l = datas2.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(len, X_l).fit()
print(model.summary())

X_l = datas2.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(len, X_l).fit()
print(model.summary())
