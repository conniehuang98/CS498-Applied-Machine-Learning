#convert y from factor to numeric
factor2numeric <- function(y){
  y<- datay
  y<- as.numeric(y)
  y[y==1] <- -1
  y[y==2] <- 1
  return(y)
}


# a: a vector of numeric of length m
# b: a numeric
# x_validate: a data frame (nrow: n, ncol: m)
# y_validate: a data frame (nrow: n, ncol: 1)

evaluate <- function(a, b, x_validate, y_validate) {
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
    else {
      if (y_i == -1.0 && gamma < 0.0) {
        accuracy <- accuracy + 1.0
      }
      else {
        next
      }
    }
  }
  accuracy <- accuracy / total
  return(accuracy)
}