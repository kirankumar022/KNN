import pandas as pd
import numpy as np

zoo=pd.read_csv("E:/Assignments/ASsignment week 12/KNN/Assignment/Zoo.csv")

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

zoo_n = norm_func(zoo.iloc[:, 1:])
zoo_n.describe()



X = np.array(zoo_n.iloc[:,:]) # Predictors 
Y = np.array(zoo['animal name']) 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 50)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred
# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions'])

# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 
# creating empty list variable 
acc = []
for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])
    
    
import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
