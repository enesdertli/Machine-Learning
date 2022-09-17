import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datas = pd.read_csv("C:\\Dosyalar\\stuff\\ders\\csv files\maaslar.csv")

x = datas.iloc[:,1:2]
y = datas.iloc[:,2:]

X = x.values
Y = y.values

#*Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#*Polynomial Regression
#*2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#*4. dereceden polinom
poly_reg3 = PolynomialFeatures(degree=4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

#*Grafikler
plt.scatter(X,Y, color="red")
plt.plot(x, lin_reg.predict(X), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

plt.scatter(X,Y, color = 'red')
plt.plot(X, lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()

#*Tahminler
print("lin reg 11 " ,lin_reg.predict([[11]]))
print("lin reg 6.6 ", lin_reg.predict([[6.6]]))

print("lin reg2 11 ",lin_reg2.predict(poly_reg.fit_transform([[11]])))
print("lin reg 2 6.6 ",lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

#*Support Vector Machine
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
#!!!!!!! Burada y yi 2 boyutlu hale getirmemiz gerekiyor. !!!!!!!
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli, y_olcekli)
plt.title("SVR")
plt.scatter(x_olcekli, y_olcekli, color = 'red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color = 'blue')
plt.show()

print("svr 11 ",svr_reg.predict([[11]]))
print("svr 6.6 ",svr_reg.predict([[6.6]]))

#*Random forest

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_reg.fit(X,Y.ravel())
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))
plt.title("Random Forest")
plt.scatter(X,Y, color = 'red')
plt.plot(X,rf_reg.predict(X), color = 'blue')
plt.show()


