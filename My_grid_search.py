# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:01:33 2019

@author: Ayush
"""

#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset
dataset= pd.read_csv('Social_Network_Ads.csv')
X= dataset.iloc[:,2:-1].values
Y= dataset.iloc[:,-1].values

#Splitting data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=0)

#Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Classifier
from sklearn.svm import SVC
classifier=SVC(kernel='rbf', random_state=0)
classifier.fit(X_train,Y_train)

#Prediction
Y_pred= classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

#K fold
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
mean=accuracies.mean()
std=accuracies.std()

#Gris search
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1,10,100], 'kernel':['linear']}, 
            {'C':[1,10,100], 'kernel':['rbf'], 'gamma':[0.5,.01,.01,.0010]}]
grid_search=GridSearchCV(estimator=classifier, param_grid=parameters, 
                         scoring='accuracy', cv=10, n_jobs=-1)
grid_search=grid_search.fit(X_train, Y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_

#Visualising the results
from matplotlib.colors import ListedColormap
X_set,Y_set=X_train,Y_train
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01), 
                    np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                    alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate (np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
    

X_set,Y_set=X_test,Y_test
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01), 
                    np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                    alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate (np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
