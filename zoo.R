library(ISLR)
library(readr)
zoo=read.csv("E:/Assignments/ASsignment week 12/KNN/Assignment/Zoo.csv")
str(zoo)
zoo$type=as.factor(zoo$type)
summary(zoo$type)
any(is.na(zoo))
zoo$animal.name=NULL
# save the Purchase column in a separate variable
purchase <- zoo[,17]

# Standarize the dataset using "scale()" R function
stzoo <- scale(zoo[,-17])
# First 100 rows for test set
test.index <- 1:20
test.data <- stzoo[test.index,]
test.purchase <- purchase[test.index]
# Rest of data for training
train.data <- stzoo[-test.index,]
train.purchase <- purchase[-test.index]
library(class)

predicted.purchase <- knn(train.data,test.data,train.purchase,k=22)
head(predicted.purchase)
mean(test.purchase != predicted.purchase)
predicted.purchase <- knn(train.data,test.data,train.purchase,k=30)
mean(test.purchase != predicted.purchase)
predicted.purchase <- knn(train.data,test.data,train.purchase,k=80)
mean(test.purchase != predicted.purchase)

predicted.purchase = NULL
error.rate = NULL

for(i in 1:20){
  
  predicted.purchase = knn(train.data,test.data,train.purchase,k=i)
  error.rate[i] = mean(test.purchase != predicted.purchase)
}
print(error.rate)
library(ggplot2)
k.values <- 1:20
error.df <- data.frame(error.rate,k.values)
error.df
ggplot(error.df,aes(x=k.values,y=error.rate)) + geom_point()+ geom_line(lty="dotted",color='red')
