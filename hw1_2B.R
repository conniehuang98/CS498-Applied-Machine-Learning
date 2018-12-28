source('Reader.R')

library(caret)
library(h2o)

h2o.init()

tr_labels <- read_label("train-labels.idx1-ubyte", 60000)
te_labels <- read_label("t10k-labels.idx1-ubyte", 10000)

#untouched
tr_ut <- read_untouched("train-images.idx3-ubyte", 60000)
df_tr_ut <- data.frame(tr_ut)
df_tr_ut$label <- tr_labels
h2o_df_tr_ut <- as.h2o(df_tr_ut)
features<-colnames(df_tr_ut)[!(colnames(df_tr_ut) %in% c("label", "Label"))]

#trainig
#10
rfut_4_10 <- h2o.randomForest(x=features, y="label", ntrees = 10, max_depth = 4, training_frame=h2o_df_tr_ut)
rfut_8_10 <- h2o.randomForest(x=features, y="label", ntrees = 10, max_depth = 8, training_frame=h2o_df_tr_ut)
rfut_16_10 <- h2o.randomForest(x=features, y="label", ntrees = 10, max_depth = 16, training_frame=h2o_df_tr_ut)

#30
rfut_4_30 <- h2o.randomForest(x=features, y="label", ntrees = 30, max_depth = 4, training_frame=h2o_df_tr_ut)
rfut_8_30 <- h2o.randomForest(x=features, y="label", ntrees = 30, max_depth = 8, training_frame=h2o_df_tr_ut)
rfut_16_30 <- h2o.randomForest(x=features, y="label", ntrees = 30, max_depth = 16, training_frame=h2o_df_tr_ut)


te_imgs_ut <- read_untouched("t10k-images.idx3-ubyte", 10000)
df_te_ut <- data.frame(te_imgs_ut)
h2o_df_te_ut <- as.h2o(df_te_ut)

#result
predictions<-as.data.frame(h2o.predict(rfut_4_10,h2o_df_te_ut))
confusionMatrix(predictions[,1], te_labels)
predictions<-as.data.frame(h2o.predict(rfut_4_30,h2o_df_te_ut))
confusionMatrix(predictions[,1], te_labels)

predictions<-as.data.frame(h2o.predict(rfut_8_10,h2o_df_te_ut))
confusionMatrix(predictions[,1], te_labels)
predictions<-as.data.frame(h2o.predict(rfut_8_30,h2o_df_te_ut))
confusionMatrix(predictions[,1], te_labels)

predictions<-as.data.frame(h2o.predict(rfut_16_10,h2o_df_te_ut))
confusionMatrix(predictions[,1], te_labels)
predictions<-as.data.frame(h2o.predict(rfut_16_30,h2o_df_te_ut))
confusionMatrix(predictions[,1], te_labels)



##################################################################################################
#bounded 
train_bounded <- bound_image(train_untouched)
df_tr_bounded <- data.frame(train_bounded)
df_tr_bounded$label <- tr_labels
h2o_df_tr_bounded <- as.h2o(df_tr_bounded)
features<-colnames(df_tr_bounded)[!(colnames(df_tr_bounded) %in% c("label", "Label"))]


#Bounded 10
rfBounded_4_10 <- h2o.randomForest(x=features, y="label", ntrees = 10, max_depth = 4, training_frame=h2o_df_tr_bounded)
rfBounded_8_10 <- h2o.randomForest(x=features, y="label", ntrees = 10, max_depth = 8, training_frame=h2o_df_tr_bounded)
rfBounded_16_10 <- h2o.randomForest(x=features, y="label", ntrees = 10, max_depth = 16, training_frame=h2o_df_tr_bounded)

#Bounded 30
rfBounded_4_30 <- h2o.randomForest(x=features, y="label", ntrees = 30, max_depth = 4, training_frame=h2o_df_tr_bounded)
rfBounded_8_30 <- h2o.randomForest(x=features, y="label", ntrees = 30, max_depth = 8, training_frame=h2o_df_tr_bounded)
rfBounded_16_30 <- h2o.randomForest(x=features, y="label", ntrees = 30, max_depth = 16, training_frame=h2o_df_tr_bounded)

te_imgs_bounded <- bound_image(te_imgs_ut)
df_te_bounded <- data.frame(te_imgs_bounded)
h2o_df_te_bounded <- as.h2o(df_te_bounded)

predictions<-as.data.frame(h2o.predict(rfBounded_4_10,h2o_df_te_bounded))
confusionMatrix(predictions[,1], te_labels)
predictions<-as.data.frame(h2o.predict(rfBounded_8_10,h2o_df_te_bounded))
confusionMatrix(predictions[,1], te_labels)
predictions<-as.data.frame(h2o.predict(rfBounded_16_10,h2o_df_te_bounded))
confusionMatrix(predictions[,1], te_labels)

predictions<-as.data.frame(h2o.predict(rfBounded_4_30,h2o_df_te_bounded))
confusionMatrix(predictions[,1], te_labels)
predictions<-as.data.frame(h2o.predict(rfBounded_8_30,h2o_df_te_bounded))
confusionMatrix(predictions[,1], te_labels)
predictions<-as.data.frame(h2o.predict(rfBounded_16_30,h2o_df_te_bounded))
confusionMatrix(predictions[,1], te_labels)
