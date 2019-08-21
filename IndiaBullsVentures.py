# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 23:09:21 2019

@author: Tejan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import csv
from sklearn.ensemble import RandomForestRegressor
dataset = pd.read_csv('IndiaBullsVentures.csv')
dataset1 = pd.read_csv('IBVenturesJul.csv')
datasetJun = pd.read_csv('IBVenturesJun.csv')

#dataset['Date'] = pd.to_datetime(dataset.Date, format = '%Y-%m-%d')
#dataset.index = dataset['Date']

#to plot Monthly Data
date = []
close = []
open1 = []
high = []
low = []

def get_data(filename):
      with open(filename, 'r') as csvfile:
            csvFileReader =   csv.reader(csvfile)
            next(csvFileReader)
            for row in csvFileReader:
                  date.append(int(row[0].split('-')[2]))
                  open1.append(float(row[1]))
                  high.append(float(row[2]))
                  low.append(float(row[3]))
                  close.append(float(row[4]))
      return
get_data('IBVenturesJul.csv')
def predict_prices(date, close, x):
      date = np.reshape(date, (len(date), 1))
      random = RandomForestRegressor(n_estimators = 1000)
      random.fit(date, close)
      decision = DecisionTreeRegressor()
      decision.fit(date, close)
      svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)
      svr_rbf.fit(date, close)
      
      plt.scatter(date, close, color = 'black', label= 'data')
      plt.plot(date, svr_rbf.predict(date), color = 'red', label = 'rbf')
      plt.plot(date, random.predict(date), color = 'blue', label = 'RandomForest')
      plt.plot(date, decision.predict(date), color = 'yellow', label = 'DecisionTree')
      plt.xlabel('Date')
      plt.ylabel('close')
      plt.legend()
      plt.show()
      
      return svr_rbf.predict(x)[0], random.predict(x)[0], decision.predict(x)[0]

predicted_price = predict_prices(date, close, 29)

dateJun = []
closeJun = []
open1Jun = []
highJun = []
lowJun = []

def get_dataJun(filename):
      with open(filename, 'r') as csvfile:
            csvFileReader =   csv.reader(csvfile)
            next(csvFileReader)
            for row in csvFileReader:
                  dateJun.append(int(row[0].split('/')[0]))
                  open1Jun.append(float(row[1]))
                  highJun.append(float(row[2]))
                  lowJun.append(float(row[3]))
                  closeJun.append(float(row[4]))
      return
get_dataJun('IBVenturesJun.csv')
def predict_pricesJun(dateJun, closeJun, x):
      dateJun = np.reshape(dateJun, (len(dateJun), 1))
      random = RandomForestRegressor(n_estimators = 1000)
      random.fit(dateJun, closeJun)
      decision = DecisionTreeRegressor()
      decision.fit(dateJun, closeJun)
      svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)
      svr_rbf.fit(dateJun, closeJun)
      
      plt.scatter(dateJun, closeJun, color = 'black', label= 'data')
      plt.plot(dateJun, svr_rbf.predict(dateJun), color = 'red', label = 'rbf')
      plt.plot(dateJun, random.predict(dateJun), color = 'blue', label = 'RandomForest')
      plt.plot(dateJun, decision.predict(dateJun), color = 'yellow', label = 'DecisionTree')
      plt.xlabel('June')
      plt.ylabel('close')
      plt.legend()
      plt.show()
      
      return svr_rbf.predict(x)[0], random.predict(x)[0], decision.predict(x)[0]

predicted_priceJun = predict_pricesJun(dateJun, closeJun, 29)



#open vs predicted for years
plt.figure(figsize = (16,8))
plt.plot(dataset['Close'], label = 'Close price history')

dataset.head()


dataset.Close.plot()
plt.show()

dataset.Open.plot()
plt.show()

dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Date']= dataset['Date'].map(dt.date.toordinal)

X = dataset.iloc[:, [0,1,2,3]].values
y = dataset.iloc[:, 4].values.reshape(-1,1)



close = dataset['Close']
close.describe()


#HeatMap
cormatt = dataset.corr()
top_corr_features = cormatt.index
plt.figure(figsize = (6,6))
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)


#Linear Regressor
from sklearn.linear_model import LinearRegression
regressorLinear = LinearRegression()
regressorLinear.fit(X_train, y_train)
y_predLinear = regressorLinear.predict(X_test)
y_predLinear = sc_y.inverse_transform(y_predLinear)




# Visualising the Training set results
"""plt.scatter(X_train, y_train, color = 'orange')
plt.plot(X_train, regressorLinear.predict(X_train), color = 'cyan')
plt.title('Opening vs Closing Bombay Dying (Training set)')
plt.xlabel('Opening')
plt.ylabel('Closing')
plt.show()
# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'orange')
plt.plot(X_train, regressorLinear.predict(X_train), color = 'cyan')
plt.title('Opening vs Closing Bombay Dying (Test set)')
plt.xlabel('Opening')
plt.ylabel('Closing')
plt.show()
"""
#SVR 
from sklearn.svm import SVR #performance very POOR
regressorSvr = SVR(kernel = 'rbf', degree = 3)
regressorSvr.fit(X_train, y_train)
y_predSVR = regressorSvr.predict(X_test)
y_predSVR = sc_y.inverse_transform(y_predSVR)


"""# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'orange')
plt.plot(X_train, regressorSvr.predict(X_train), color = 'cyan')
plt.title('Opening vs Closing Bombay Dying (Training set)')
plt.xlabel('Opening')
plt.ylabel('Closing')
plt.show()
# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'orange')
plt.plot(X_train, regressorSvr.predict(X_train), color = 'cyan')
plt.title('Opening vs Closing Bombay Dying (Test set)')
plt.xlabel('Opening')
plt.ylabel('Closing')
plt.show()
"""
#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
regressorDecision = DecisionTreeRegressor(random_state = 0)
regressorDecision.fit(X_train, y_train)
y_predDecision = regressorDecision.predict(X_test)
y_predDecision = sc_y.inverse_transform(y_predDecision)


"""#visualising training set
X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, regressorDecision.predict(X_grid), color = 'blue')
plt.title('Opeining vs Closing Bombay Dying')
plt.xlabel('Opeing')
plt.ylabel('Closing')
plt.show()
#visualising test set
X_grid = np.arange(min(X_test), max(X_test), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_grid, regressorDecision.predict(X_grid), color = 'blue')
plt.title('Opeining vs Closing Bombay Dying')
plt.xlabel('Opening')
plt.ylabel('Closing')
plt.show()
"""
#Random forest tree
#from sklearn.ensemble import RandomForestRegressor
regressorRandom = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressorRandom.fit(X_train, y_train)
y_predRandom = regressorRandom.predict(X_test)
y_predRandom = sc_y.inverse_transform(y_predRandom)


"""#visualising training set
X_grid = np.arange(min(X_train), max(X_train), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'orange')
plt.plot(X_grid, regressorRandom.predict(X_grid), color = 'cyan')
plt.title('Opeining vs Closing Bombay Dying')
plt.xlabel('Opeing')
plt.ylabel('Closing')
plt.show()
#visualising test set
X_grid = np.arange(min(X_test), max(X_test), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'orange')
plt.plot(X_grid, regressorRandom.predict(X_grid), color = 'cyan')
plt.title('Opeining vs Closing Bombay Dying')
plt.xlabel('Opening')
plt.ylabel('Closing')
plt.show()
"""
X_test = sc_X.inverse_transform(X_test)


dict = {'Date' : X_test[:,0], 'Open' : X_test[:,1], 'High' : X_test[:,2], 'Low' : X_test[:,3], 'Actual' : y_test[:,0], 'Linear' : y_predLinear[:,0], 'SVR' : y_predSVR, 'Decision': y_predDecision, 'RandomForest': y_predRandom }
DataFramePredicted = pd.DataFrame(dict)
#DataFramePredicted['Date'] = DataFramePredicted['Date'].map(dt.date.fromordinal)
#DataFramePredicted.to_csv('IndiaBullsVenturesPred.csv')


