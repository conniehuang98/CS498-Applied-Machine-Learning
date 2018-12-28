setwd("~/Documents/AML_hw3")
library(FactoMineR)

###### reference https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com/229093#229093?newreg=37fd96da1eb4484dad8677d255392011

######## Read data from file ###############
dataI <- read.csv("hw3-data/dataI.csv", header = TRUE)
dataII <- read.csv("hw3-data/dataII.csv", header = TRUE)
dataIII <- read.csv("hw3-data/dataIII.csv", header = TRUE)
dataIV <- read.csv("hw3-data/dataIV.csv", header = TRUE)
dataV <- read.csv("hw3-data/dataV.csv", header = TRUE)
iris <- read.csv("hw3-data/iris.csv", header = TRUE)


################################################# N cols ##############################################  
####################### reconstruction
Eigenvectors <- eigen(cov(iris))$vectors

X = dataV #switch data here
mu = colMeans(iris)

Xpca = prcomp(X)

nComp = 4
Xhat = Xpca$x[,1:nComp] %*% t(Eigenvectors[,1:nComp])
Xhat = scale(Xhat, center = -mu, scale = FALSE)

#write.csv(Xhat, "hwu63-recon.csv", row.names = FALSE)
Xhat[1,] #first element

matrix<-(Xhat-iris)^2
mse <- sum(matrix)/(150*4)
print(mse)

################## for 0N cols
N0 = colMeans(iris, na.rm = FALSE, dims = 1)
print(N0)
mse0 <- (sum((iris[,1]-N0[1])^2) + sum((iris[,2]-N0[2])^2) + 
           sum((iris[,3]-N0[3])^2) + sum((iris[,4]-N0[4])^2)) /(150*4)
print(mse0)


################################################# C cols ###############################################  
####################### reconstruction
X = iris #switch data here
mu = colMeans(X)

Xpca = prcomp(X)

nComp = 1
Xhat = Xpca$x[,1:nComp] %*% t(Xpca$rotation[,1:nComp])
Xhat = scale(Xhat, center = -mu, scale = FALSE)

#write.csv(Xhat, "hwu63-recon.csv", row.names = FALSE)
Xhat[1,] #first element

################## mse
matrix<-(Xhat-iris)^2
mse <- sum(matrix)/(150*4)
print(mse)

################## for 0c cols
C0 = colMeans(X, na.rm = FALSE, dims = 1)
print(C0)
mse0 <- (sum((iris[,1]-C0[1])^2) + sum((iris[,2]-C0[2])^2) + 
           sum((iris[,3]-C0[3])^2) + sum((iris[,4]-C0[4])^2)) /(150*4)
print(mse0)



