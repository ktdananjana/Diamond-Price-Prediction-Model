# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 10:13:56 2018

@author: Tharuka
"""

# Importing useful libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as pre
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 
from sklearn.neural_network import MLPRegressor
import seaborn as sns

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# importing data
data_path= 'D:\Hackstat\diamonds.csv'
diamonds = pd.read_csv(data_path)
diamonds.head()
diamonds.drop(['ID'], axis=1, inplace=True)
diamonds.shape
diamonds.info()
diamonds.describe()
diamonds = diamonds[(diamonds[['x','y','z']] != 0).all(axis=1)]
diamonds.loc[(diamonds['x']==0) | (diamonds['y']==0) | (diamonds['z']==0)]

# Correlation Map
corr = diamonds.corr()
sns.heatmap(data=corr, square=True , annot=True, cbar=True)
plt.figure(figsize=[12,12])

# Price vs Catat
sns.jointplot(x='carat' , y='price' , data=diamonds , size=5)

# Price vs Cut
sns.factorplot(x='cut', y='price', data=diamonds, kind='box' ,aspect=2.5 )

# price vs color
sns.factorplot(x='color', y='price' , data=diamonds , kind='violin', aspect=2.5)

# Clarity vs price

labels = diamonds.clarity.unique().tolist()
sizes = diamonds.clarity.value_counts().tolist()
colors = ['#006400', '#E40E00', '#A00994', '#613205', '#FFED0D', '#16F5A7','#ff9999','#66b3ff']
explode = (0.1, 0.0, 0.1, 0, 0.1, 0, 0.1,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=0)
plt.axis('equal')
plt.title("Percentage of Clarity Categories")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(6,6)
plt.show()
sns.boxplot(x='clarity', y='price', data=diamonds )

# Depth vs Price
sns.jointplot(x='depth', y='price' , data=diamonds , kind='regplot', size=5)

# Table vs Price
sns.jointplot(x='table', y='price', data=diamonds , size=5)

#price vs volume
diamonds['volume'] = diamonds['x']*diamonds['y']*diamonds['z']
diamonds.head()
sns.jointplot(x='volume', y='price' , data=diamonds, size=5)


# Model to find the best regression model


data_path= 'D:\Hackstat\diamonds.csv'
model_data = pd.read_csv(data_path)
model_data.head()
model_data.drop(['ID'], axis=1, inplace=True)

print(model_data['cut'].unique())
print(model_data['color'].unique())
print(model_data['clarity'].unique())
model_data['cut'].head()


model_data = pd.concat([model_data, pd.get_dummies(model_data['cut'], prefix='cut', drop_first=True)],axis=1)
model_data = pd.concat([model_data, pd.get_dummies(model_data['color'], prefix='color', drop_first=True)],axis=1)
model_data = pd.concat([model_data, pd.get_dummies(model_data['clarity'], prefix='clarity', drop_first=True)],axis=1)
model_data.drop(['cut','color','clarity'], axis=1, inplace=True)
model_data.head()
cols = list(model_data)
print(cols)
model_data.plot.scatter(x='carat', y='price', s=1);


model_data['carat_squared'] = model_data['carat']**2

target_name = 'price'
#scaler = sk.preprocessing
robust_scaler = pre.RobustScaler() 
X = model_data.drop('price', axis=1)
X = robust_scaler.fit_transform(X)
y = model_data[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['NULL', 'MLR', 'KNN', 'LASSO'])

#Null Model
y_pred_null = y_train.mean()
models.loc['train_mse','NULL'] = mean_squared_error(y_pred=np.repeat(y_pred_null, y_train.size), 
                                                    y_true=y_train)
models.loc['test_mse','NULL'] = mean_squared_error(y_pred=np.repeat(y_pred_null, y_test.size), 
                                                   y_true=y_test)

#Linear Regression
linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

models.loc['train_mse','MLR'] = mean_squared_error(y_pred=linear_regression.predict(X_train), 
                                                    y_true=y_train)

models.loc['test_mse','MLR'] = mean_squared_error(y_pred=linear_regression.predict(X_test), 
                                                   y_true=y_test)

#Ridge Regreesion
ridge_regression = Ridge()

ridge_regression.fit(X_train, y_train)

models.loc['train_mse','Ridge'] = mean_squared_error(y_pred=ridge_regression.predict(X_train), 
                                                    y_true=y_train)

models.loc['test_mse','Ridge'] = mean_squared_error(y_pred=ridge_regression.predict(X_test), 
                                                   y_true=y_test)




#KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10, weights='distance', metric='euclidean', n_jobs=-1)

knn.fit(X_train, y_train)

models.loc['train_mse','KNN'] = mean_squared_error(y_pred=knn.predict(X_train), 
                                                    y_true=y_train)

models.loc['test_mse','KNN'] = mean_squared_error(y_pred=knn.predict(X_test), 
                                                   y_true=y_test)


# Lasso Regression
lasso = Lasso(alpha=0.1)

lasso.fit(X_train, y_train)

models.loc['train_mse','LASSO'] = mean_squared_error(y_pred=lasso.predict(X_train), 
                                                    y_true=y_train)

models.loc['test_mse','LASSO'] = mean_squared_error(y_pred=lasso.predict(X_test), 
                                                   y_true=y_test)




# Random Forest Regressor

clf_rf = RandomForestRegressor()

clf_rf.fit(X_train, y_train)

models.loc['train_mse','RF'] = mean_squared_error(y_pred=clf_rf.predict(X_train), 
                                                    y_true=y_train)

models.loc['test_mse','RF'] = mean_squared_error(y_pred=clf_rf.predict(X_test), 
                                                   y_true=y_test)



# Make Models of data To the best regressor
fig, ax = plt.subplots(figsize=(8,5))
models.loc['test_mse'].plot(kind='barh', ax=ax)
ax.set_title('Test MSE for Regression Models')
ax.legend(loc=8, ncol=4);


fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(knn.predict(X_test), y_test, s=4)
ax.plot(y_test, y_test, color='red')
ax.set_title('KNN: predictions vs. observed values (test data)')
ax.set_xlabel('Predicted prices')
ax.set_ylabel('Observed prices');
diamonds.head()


rf_final = RandomForestRegressor(n_estimators =50, n_jobs=-1)

rf_final.fit(X, y)


new_diamond = OrderedDict([('carat',0.45), ('depth',62.3), ('table',59.0), ('x',3.95),
                           ('y',3.92), ('z',2.45), ('cut_Good',0.0), ('cut_Ideal',0.0),
                           ('cut_Premium',1.0), ('cut_Very Good',0.0), ('color_E',0.0), 
                           ('color_F',0.0), ('color_G',1.0), ('color_H',0.0), ('color_I',0.0),
                           ('color_J',0.0), ('clarity_IF',0.0), ('clarity_SI1',0.0),
                           ('clarity_SI2',0.0), ('clarity_VS1',0.0), ('clarity_VS2',0.0),
                           ('clarity_VVS1',1.0), ('clarity_VVS2',0.0), ('carat_squared',0.0576)])

new_diamond = pd.Series(new_diamond).values.reshape(1,-1)


rf_final.predict(new_diamond)