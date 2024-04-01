
#import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#import numpy and pandas libraries 
import numpy as np
import pandas as pd

#read iris flower data set and assign it to pandas dataframe
df_iris = pd.read_csv('https://raw.githubusercontent.com/mpourhoma/CS4661/master/iris.csv')

#create feature matrix for iris dataset:
#this creates a list of feature names in the dataset
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#select features from the DataFrame
X = df_iris[feature_cols]

#printing list of features for checking purposes
X[::10]

#selects labels from DataFrame and set in y variable
y = df_iris['species']

#print list of labels for checking purposes
y[::10]

#import train_test_split library
#split data into training (60%) and testing set (40%)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4,random_state = 6)

#Instantiate object of KNeighborsClassifier with k = 3
k = 3
knn_kevin = KNeighborsClassifier(n_neighbors = k)

#train the model on training set
knn_kevin.fit(X_train, y_train)

#test model on testing set
y_predict = knn_kevin.predict(X_test)
print(y_predict)

#import accuracy score and find accuracy percentage
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_predict)
print(accuracy)

#save K values into a list
list = [1, 5, 7, 11, 15, 27, 59]
#create empty list to save accuracy values in
acc = []
#create forloop to iterate through list and find accuracy based on "K" values in list

for l in list:
    #set n_neighbors to values in list
    knn_kevin = KNeighborsClassifier(n_neighbors=l)
    #train data on training set
    knn_kevin.fit(X_train, y_train)
    #test data on testing set
    y_predict = knn_kevin.predict(X_test)
    #find accuracy of testing set vs actual lavels
    accuracy = accuracy_score(y_test, y_predict)
    #save the accuracy values into empty list "acc"
    acc.append(accuracy)
    
print(acc)
    
