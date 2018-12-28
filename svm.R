setwd('~/Documents/AML_hw2/')
library(klaR)
library(caret)
library(pracma)
library(matrixStats)

#convert y from factor to numeric
factor2numeric <- function(y){
  y<- datay
  y<- as.numeric(y)
  y[y==1] <- -1
  y[y==2] <- 1
  return(y)
}

calc_accuracy <- function(a, b, x_validate, y_validate) {
  total <- 0.0
  accuracy <- 0.0
  validate_row_n <- nrow(x_validate)
  row.names(x_validate) <- c(1:validate_row_n)
  for (i in 1:validate_row_n) {
    x_i = as.numeric(x_validate[i,])
    y_i = as.numeric(y_validate[i])
    gamma <- dot(a,x_i) + b
    total <- total + 1.0
    if (y_i == 1.0 && gamma >= 0.0) {
      accuracy <- accuracy + 1.0
    }
    else if (y_i == -1.0 && gamma < 0.0) {
        accuracy <- accuracy + 1.0
      }
    else {
        next
      }
    }
  accuracy <- accuracy / total
  return(accuracy)
}

#Preprocess Data  ///////////////////////////////

#test
test_x<-read.csv('test.data', header=FALSE, sep= ",")
test_x<-subset(test_x,select=c(1,3,5,11,12,13))
test_x <- na.omit(test_x)
test_xmean<-sapply(test_x , mean, na.rm=TRUE) #mean
test_xsd<-sapply(test_x , sd, na.rm=TRUE) #sd
test_xoffsets<-t(t(test_x)-test_xmean)
test_x <-t(t(test_xoffsets)/test_xsd)

test_y<-data.frame( Label=rep(NA, nrow(test_x)), num = rep(NA, nrow(test_x)),stringsAsFactors=FALSE) 
row.names(test_y) = 1:nrow(test_x)

#train data
data<-read.csv('train.data', header=FALSE, sep= ",")
data <-subset(data,select=c(1,3,5,11,12,13,15))
data <- na.omit(data)
datax<-data[,-ncol(data)]
datay<-data[,ncol(data)]
datay <-factor2numeric(datay)

dataxmean<-sapply(datax, mean, na.rm=TRUE) #mean
dataxsd<-sapply(datax, sd, na.rm=TRUE) #sd
dataxoffsets<-t(t(datax)-dataxmean)
datax<-t(t(dataxoffsets)/dataxsd)

train<-createDataPartition(y=datay, p=.9, list=FALSE)
validation_x<-datax[-train,]
validation_y<- datay[-train]

#train data
train_x <- datax[train,]
train_y <- datay[train]


# Define Constants  ///////////////////////////////
max_epoch = 100  
m = 1         
n = 10         
Ns = 300      
k = 30        

validate_accuracy <- vector(mode="numeric", length=4)
#test_accuracy <-vector(mode="numeric", length=4)
lambda <- c(0.001, 0.01, 0.1, 1)
#m_vec <- c(3, 9, 27, 81)
#n_vec <- c(0.333, 1, 3, 9)


# Begin Training ///////////////////////////////
for(i in 1:4)  
{
  # initial model
  a = c(0,0,0,0,0,0) 
  b = 0  
  
  plot_accuracy <- vector(mode="numeric", length=1000)
  mag_coefficient <- vector(mode="numeric", length=1000)
 
  
  for(s in 1:max_epoch) 
  {
    step_length <- m/(0.01*s + n)
    
    train_x_row <-sample(nrow(train_x),50, replace= FALSE)
    train_x_train <- train_x[-train_x_row,]
    train_y_train <- train_y[-train_x_row]
    
    validation_x_train <- train_x[train_x_row,]
    validation_y_train <- train_y[train_x_row]
    
    for(j in 1:Ns)
    {
      data_point_row <- sample(nrow(train_x_train),1, replace= FALSE)
      data_point_x <- train_x_train[data_point_row,]
      data_point_y <- train_y_train[data_point_row]
      
      if((dot(a,as.numeric(data_point_x))+b)*data_point_y >= 1){
        a = a - step_length*lambda[i]*a
        b = b
      }else{
        a = a -step_length*(lambda[i]*a - data_point_y*data_point_x)
        b = b + step_length*data_point_y
      }
      
      if(mod(j,k)==0){
        
        plot_accuracy[Ns*(s-1)/k+j%/%30] = calc_accuracy(a,b,validation_x_train,validation_y_train)
        mag_coefficient[Ns*(s-1)/k+j%/%30] = sqrt(dot(a,a))
      }
    }
    
  }
  
  #plot(plot_accuracy, type = "o",  ylim=range(0,1), ylab="Accuracy", xlab="Epochs",  col = c("red", "blue"), main = "Plot_accuracy_step4")
  #plot(mag_coefficient, type="o", ylim=range(0, mag_coefficient), xlab="Epochs", ylab="Mag_coefficient",  col = c("red", "blue"), main = "Plot_magco_step4")
  #validate_accuracy[i] = calc_accuracy(a,b,validation_x, validation_y)
  #print(validate_accuracy[i])
  
}

for(j in 2:nrow(test_x)){
  x_i <- as.numeric(test_x[j,])
  test_y$num[j-1]<- dot(a, x_i) + b
}
test_y$num<-scale(test_y$num)
for(j in 2:nrow(test_x)){
  if(test_y$num[j-1] >= 0.5){
    test_y$Label[j-1] <- ">50K"
  }else{
    test_y$Label[j-1] <- "<=50K"
  }
}

write.csv(test_y$Label, file = "hwu63_hw2.csv")

