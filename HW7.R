library(glmnet)
library(readr)
# -----------------12.3--------------------
blogData <- read_csv("~/Downloads/BlogFeedback/blogData_train.csv", col_names = FALSE)
X <- as.matrix(blogData[,c(1:280)])
Y <- as.matrix(blogData[,281])
cv <- cv.glmnet(X, Y, family="poisson")
plot(cv)

p1 <- predict(cv, X, s='lambda.1se')
p2 <- predict(cv, X, s='lambda.min')
plot(Y, p1)
plot(Y, p2)

setwd("~/Downloads/BlogFeedback/")
testfiles <- list.files(pattern = "*00.csv")
testdata <- lapply(testfiles, function(i){
  read.csv(i, header=FALSE)
})
library(data.table)
testdata <- rbindlist(testdata)
tX <- as.matrix(testdata[,c(1:280)])
tY <- as.matrix(testdata[,281])
pt1 <- predict(cv, tX, s='lambda.1se')
pt2 <- predict(cv, tX, s='lambda.min')
plot(tY, pt1)
plot(tY, pt2)
# ----------------12.4-------------------
genedata <- read.table("~/Desktop/CS498AML/HW7/gene.txt", quote = "\"")
tissues <- read.table("~/Desktop/CS498AML/HW7/tissues.txt", quote="\"")
genedata <- as.matrix(t(genedata))
tissues[tissues >0 ] = 1
tissues[tissues <0] = 0
table(tissues)
tissues <- as.matrix(tissues)
logis <- cv.glmnet(genedata, tissues, type.measure = "class", family="binomial")
plot(logis)
blacc <- 40/62
print(blacc)
results <- predict(logis, genedata, s="lambda.min")
results[results > mean(results)] = 1
results[results < mean(results)] = 0
mean(results==tissues)
#------------------12.5------------------
micedata <- read_csv("~/Downloads/Crusio1.csv", col_names = TRUE)
micedata <- micedata[,c(2,4:41)]
micedata <- micedata[complete.cases(micedata),]
micedataX <- as.matrix(micedata[,c(2:39)])
micedataY <- as.matrix(micedata[,1])

micedataY[micedataY == 'f'] = 0
micedataY[micedataY == 'm'] = 1
table(micedataY)
logislasso <- cv.glmnet(micedataX, micedataY, type.measure = "class", alpha = 1, family="binomial")
plot(logislasso)
gender <- predict(logislasso, micedataX)
gender <- ifelse(gender > mean(gender), 1, 0)
mean(gender==micedataY)

print(logislasso$lambda.min)

micedata <- read_csv("~/Downloads/Crusio1.csv", col_names = TRUE)
micedata <- micedata[,c(1,4:41)]
micedata <- micedata[complete.cases(micedata),]
micedata <- micedata[micedata$strain %in% names(table(micedata$strain))[table(micedata$strain) > 9], ]

micedataY <- as.matrix(micedata[,1])
micedataX <- as.matrix(micedata[,c(2:39)])
micedataY <- as.matrix(micedataY[micedataY %in% names(table(micedataY))[table(micedataY) > 9], ])

logislasso <- cv.glmnet(micedataX, micedataY, type.measure = "class", alpha= 1, family="multinomial")
table(micedataY)
strain <- predict(logislasso, micedataX, s="lambda.1se", type="response")
head(strain)

