# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:01:52 2019

@author: hari4
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv("Churn_Modelling.csv")
x_mtx = dataset.iloc[:, 3:-1].values
y_vctr = dataset.iloc[:, -1].values

#Converting categorical data into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x_1 = LabelEncoder()
x_mtx[:,1] = labelencoder_x_1.fit_transform(x_mtx[:,1])

labelencoder_x_2 = LabelEncoder()
x_mtx[:,2] = labelencoder_x_2.fit_transform(x_mtx[:,2])

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x_mtx = np.array(ct.fit_transform(x_mtx), dtype=np.float)

#Avoiding the dummy variable trap
x_mtx = x_mtx[:, 1:]

#Splitting data into training set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_mtx, y_vctr, test_size=0.20, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train =  sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Initializing the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#Adding the input nodes and first Hidden layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))

#Adding the next hidden layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))

#Adding the output layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

#Compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size=2, epochs=100)

y_prdc = classifier.predict(x_test)
y_prdc = [1 if val > 0.5 else 0 for val in y_prdc]

from sklearn.metrics import confusion_matrix, classification

cm = confusion_matrix(y_test, y_prdc)

acc= classification.accuracy_score(y_test, y_prdc)
