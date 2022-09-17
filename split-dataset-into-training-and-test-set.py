from cgitb import reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


datas = pd.read_csv("eksikveriler.csv")

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

age = datas.iloc[:,1:4].values
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])


country = datas.iloc[:,0:1].values
print(country)

le = preprocessing.LabelEncoder()

country[:,0] = le.fit_transform(country[:,0])
print(country)

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country) 

result = pd.DataFrame(data=country, index= range(22), columns = ['fr','tr','us'])
print(result)

result2 = pd.DataFrame(data=age, index= range(22), columns = ['boy','kilo','yas'])

sex = datas.iloc[:,-1].values

result3 = pd.DataFrame(data = sex, index= range(22), columns = ['cinsiyet'])

s = pd.concat([result, result2], axis=1)
print(s)
s2 = pd.concat([s, result3], axis= 1)
print(s2)

x_train, x_test, y_train, y_test = train_test_split(s,result3,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
