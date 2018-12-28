library(readr)
source('Reader.R')
library(naivebayes)
library(caret)

#Gaussian

#untouched
#train_untouched <- read_untouched("train-images.idx3-ubyte", 60000)
#train_labels <- read_label("train-labels.idx1-ubyte", 60000)
#test_untouched <- read_untouched("t10k-images.idx3-ubyte", 10000)
#test_labels <- read_label("t10k-labels.idx1-ubyte", 10000)

#sapply(train_untouched[train_labels == 0,], mean, na.rm = T) 

train <- read_csv("train.csv")
test_untouched <- read_csv("test.csv")
train_untouched <- train[,-c(1)]
train_labels <- train[,1]

model_untouched_g <- naive_bayes(train_untouched, train_labels)
result_untouched_g <- predict(model_untouched_g, test_untouched)
confusionMatrix(result_untouched_g, test_labels)
#dim(test_untouched)
#data<-table(ImageId=[1:10000], Label= result_untouched_g)
#write.table(data, "hwu63_1.csv", sep = ",", col.names = T)

#bounded
train_bounded <- bound_image(train_untouched)
test_bounded <- bound_image(test_untouched)

model_bounded_g <- naive_bayes(train_bounded, train_labels)
result_bounded_g <- predict(model_bounded_g, test_bounded)
confusionMatrix(result_bounded_g, test_labels)

#bernoulli

#untouched 
train_untouched[,][train_untouched[,] < 127] <- 0
train_untouched[,][train_untouched[,] >= 127] <- 1
for (i in 1:(28*28)) {
  train_untouched[, i] <- factor(train_untouched[, i], levels=c(0,1))
}

test_untouched[,][test_untouched[,] < 127] <- 0
test_untouched[,][test_untouched[,] >= 127] <- 1
for (i in 1:(28*28)) {
  test_untouched[, i] <- factor(test_untouched[, i], levels=c(0,1))
}

model_untouched_b <- naive_bayes(train_untouched, train_labels, laplace=1)
result_untouched_b <- predict(model_untouched_b, test_untouched)
confusionMatrix(result_untouched_b, test_labels)


#bounded 
train_bounded[,][train_bounded[,] < 127] <- 0
train_bounded[,][train_bounded[,] >= 127] <- 1
for (i in 1:(20*20)) {
  train_bounded[,i] <- factor(train_bounded[,i], levels = c(0,1))
}

test_bounded[,][test_bounded[,] < 127] <- 0
test_bounded[,][test_bounded[,] >= 127] <- 1
for (i in 1:(20*20)) {
  test_bounded[,i] <- factor(test_bounded[,i], levels = c(0,1))
}

model_bounded_b <- naive_bayes(train_bounded, train_labels, laplace=1)
result_bounded_b <- predict(model_bounded_b, test_bounded)
confusionMatrix(result_bounded_b, test_labels)
