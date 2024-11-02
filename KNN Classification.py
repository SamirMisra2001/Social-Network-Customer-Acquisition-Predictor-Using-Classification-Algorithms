# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:02:08 2024

@author: mishr
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

dataset = pd.read_csv(r"C:\Users\mishr\Desktop\Naresh i\OCTOBER\16th - KNN\16th - KNN\Social_Network_Ads.csv")

x = dataset.iloc[:,[2,3]].values
y = dataset['Purchased'].values

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(x_train,y_train)


y_pred = classifier.predict(x_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) 

sns.heatmap(cm,annot = True)

from sklearn.metrics import accuracy_score
AS = accuracy_score(y_test, y_pred) 
print(AS)


from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)                    
            
bias = classifier.score(x_train, y_train) 
print(bias)

variance = classifier.score(x_test,y_test)
print(variance)         