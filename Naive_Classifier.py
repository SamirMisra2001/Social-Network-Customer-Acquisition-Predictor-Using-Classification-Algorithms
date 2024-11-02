# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:25:50 2024

@author: mishr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv(r"C:\Users\mishr\Desktop\Naresh i\OCTOBER\17th - NAIVE BAYES\17th - NAIVE BAYES\Social_Network_Ads.csv")

x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
mnb = GaussianNB()
mnb.fit(x_train, y_train)

y_pred = mnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)