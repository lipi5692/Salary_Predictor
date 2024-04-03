# -*- coding: utf-8 -*-
"""Salary Machine Learning Model

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gsx4BtLe-egjWwGX2YNOmWB2bMUxgaH5
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[ : , :-1].values
Y= dataset.iloc [ : ,-1].values

dataset

X_train

X_test

Y_train

Y_test

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

a=list(Y_pred)

b=[]
for i in X_test:
  b.append(float(i))
print(b)

d = {"Year of Experience" : b ,"Predicted Salary": a,"Real Slaary" : Y_test}

df = pd.DataFrame(d)

df

Y_test

plt.scatter(X_train, Y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')



plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()