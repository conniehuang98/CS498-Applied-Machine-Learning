library(car)
library(MASS)
library(pracma)
Data <- read.table("~/Documents/AML_hw6/Boston_Housing_data", quote="\"", comment.char="")
linearMod <- lm(V14 ~ ., data=Data)
summary(linearMod)
#leveragePlots(linearMod) 
plot(rstandard(linearMod))

r_data <- Data[-c(365,366,369,372,373, 368, 370, 371, 413),]
row.names(r_data) <- 1:nrow(r_data)
fit <- lm(V14 ~ ., data= r_data)
summary(fit)
cd <- cooks.distance(fit)
cd <- data.frame(1:nrow(r_data), cd)
colnames(cd) <- c("idx", "value")
plot(cd$idx ~ cd$value)
text(cd$idx ~ cd$value, labels=idx, data=cd, cex=0.9, font=2)

bc <- boxcox(V14 ~ .,  data=r_data)
lambda <- bc$x[which.max(bc$y)]

new_data <- r_data
new_data$V14 <- new_data$V14 ** lambda
new_fit <- lm(V14 ~ ., data=new_data)
summary(new_fit)
plot(rstandard(new_fit))
predicted <- fitted(new_fit) ** (1/lambda)

plot(predicted ~ r_data$V14)

