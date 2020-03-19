# importing libraries

import warnings
warnings.filterwarnings('ignore')
import numpy as numpy
import pandas as pd
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras import optimizers, regularizers
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# loading the dataset
dataset = pd.read_csv(
    'C:\\Users\\Esrat Maria\\Desktop\\petrol_consumption.csv')
dataset.info()
dataset.head()
print(dataset.shape)

print(dataset['Petrol_Consumption'].value_counts())

dataset['Petrol_Consumption'].value_counts().plot(kind='bar', figsize=(15, 8))
plt.xlabel('Petrol Consumption')
plt.ylabel('count')
plt.show()

x = dataset.iloc[:, [0, 1, 2, 3]]
y = dataset.iloc[:, 4]
print(x)

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

regressor = RandomForestRegressor(
    n_estimators=20, random_state=0)
regressor.fit(X_train, Y_train)
# print(regressor.score(X_test, Y_test) * 100)
result = regressor.predict(X_test)
df = pd.DataFrame({'Actual': Y_test, 'Predict': result})
print(df)

df.plot(kind='bar', figsize=(5, 5))
plt.show()
