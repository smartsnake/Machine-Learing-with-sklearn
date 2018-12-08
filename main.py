import numpy as numpy
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("train.csv").values

testData = pd.read_csv("test.csv").values

clf = MLPClassifier(hidden_layer_sizes=(100, ), alpha = 0.0001, activation='relu')

#Training data
xtraining = data[0:21000,1:]
train_label = data[0:21000,0]

clf.fit(xtraining, train_label)

#Testing data
xtest = data[0:21000,1:]
actual_label = data[0:21000,0]

p = clf.predict(xtest)

count = 0
for i in range(0,21000):
    count+=1 if p[i] == actual_label[i] else 0
print("accuracy: ", (count/21000)*100)