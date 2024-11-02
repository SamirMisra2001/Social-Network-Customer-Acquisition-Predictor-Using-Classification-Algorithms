# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:07:39 2024

@author: mishr
"""
#Importing required libaries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


#Reading dataset
dataset = pd.read_csv(r"C:\Users\mishr\Desktop\Naresh i\OCTOBER\15th - SVM\Social_Network_Ads.csv")


#Selecting our Independent and Dependent variable
x = dataset.iloc[:,[2,3]].values
y = dataset['Purchased'].values


#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.20,random_state=0)

#Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train =   sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#Building the Support-Vector Classifier Model
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(x_train, y_train)

#Predicting the values
y_pred = classifier.predict(x_test)


#Creating the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix",cm)

bias = classifier.score(x_train,y_train)
print("Bias :",bias) 

variance = classifier.score(x_test,y_test)
print("Variance :",variance)

# To get the model's classification report 
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr