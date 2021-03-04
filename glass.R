library(ISLR)
glass=read.csv("E:/Assignments/ASsignment week 12/KNN/Assignment/glass.csv")
str(glass)
glass$Type=as.factor(glass$Type)
summary(glass$Type)
purchase=glass[,10]
stglass=scale(glass[,-10])
# First 100 rows for test set
test.index <- 1:40
test.data <- stglass[test.index,]
test.purchase <- purchase[test.index]
# Rest of data for training
train.data <- stglass[-test.index,]
train.purchase <- purchase[-test.index]
library(class)
set.seed(101)
predicted.purchase <- knn(train.data,test.data,train.purchase,k=10)
head(predicted.purchase)
mean(test.purchase != predicted.purchase)
predicted.purchase <- knn(train.data,test.data,train.purchase,k=3)
mean(test.purchase != predicted.purchase)
predicted.purchase <- knn(train.data,test.data,train.purchase,k=5)
mean(test.purchase != predicted.purchase)
