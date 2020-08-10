# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load the dataset
data = pd.read_csv('winequality-white.csv', sep=';')

# check missing data
print(data.isna().sum())

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(10, 20)
sns.boxplot(data=data, orient='v', ax=axes[0])
sns.boxplot(data=data, y='quality', orient='pH', ax=axes[1])
plt.show()

# correlation analysis
corrMatt = data.corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
sns.heatmap(corrMatt, mask=mask,  vmax=0.8, square=True, annot=True)

# Lets build the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

print(data.head())
# features (independent)
X = data.iloc[:, :-1]
# target (dependent)
y = data.iloc[:, -1]

# add bias constant
X = np.append(arr = np.ones((X.shape[0], 1)), values=X, axis=1)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

# Linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# evaluate
from sklearn.metrics import r2_score
model_r2_score = r2_score(y_test, predictions)