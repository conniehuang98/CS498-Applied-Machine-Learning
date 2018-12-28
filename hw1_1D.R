#reference to professor's code on course website

wdat<-read.csv('pima-indians-diabetes.csv', header=FALSE)
library(klaR)
library(caret)

bigx<-wdat[,-c(9)]
bigy<-as.factor(wdat[,9])
wtd<-createDataPartition(y=bigy, p=.8, list=FALSE)

svm<-svmlight(bigx[wtd,], bigy[wtd], pathsvm='C:/Users/Sharo/Desktop/aml_hw1/svm_light/')
labels<-predict(svm, bigx[-wtd,])
foo<-labels$class
result<-sum(foo==bigy[-wtd])/(sum(foo==bigy[-wtd])+sum(!(foo==bigy[-wtd])))
print(c("Accuracy: ", result))