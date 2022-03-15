# Libraries
library(FactoMineR)
library(keras)
library(ggplot2)
library(caret)

FLAGS <- flags(
  flag_numeric("drop", 0.0),
  flag_numeric ("regul", 0.0),
  flag_numeric("l3", 0)
)




data <- read.csv('parkinsons_updrs.data')
head(data)
# A glimpse to the dataframe dimensions
dim(data)
# Are there missing values?
sum(is.na(data))
summary(data)
data$severity <- ifelse(data$total_UPDRS > 25, 1, 0)

tt <- table(data$subject.)

bb <- barplot(tt, main= "# observations by subject", xlab = "subject", col=as.numeric(data$age))
text(bb,tt-4,labels=tt,cex=0.6)

# table(as.factor(data[, "subject."]))
windows()
par(mfrow=c(1,2))
boxplot(data$total_UPDRS, main = "total_UPDRS")
boxplot(data$motor_UPDRS, main = "motor_UPDRS")
par(mfrow=c(1,1))

data2 <- data
data2$subject. <- as.character(data2$subject.)

park_pca <- PCA(data2[c(1,7:22)], quali.sup = 1, graph = F)
 
windows()
plot.PCA(park_pca, axes=c(1, 2), choix="ind",
         habillage=1, col.quali="magenta", label=c("quali"),new.plot=TRUE)

plot.PCA(park_pca, axes=c(1, 2), choix="var")


# We keep only the 16 explicative variables the we want to include in the algorithm:
dataNN <- data[7:23]

# Now, we split the data in train and test
spec = c(train = .70, test = .3)
g = sample(cut(
  seq(nrow(dataNN)), 
  nrow(dataNN)*cumsum(c(0,spec)),
  labels = names(spec)
))
res <- split(dataNN, g)

X_train <- scale(res$train[1:16]) # Escalem les variables de training
X_train <- array_reshape(X_train, c(nrow(X_train), 16)) # Reshape
# X_train <- (data.matrix(X_train))
X_train <- as.matrix(X_train) # Convertim X en matrix
Y_train <- as.numeric(as.matrix(res$train[17])) # Convertim Y en variable binària 1/0.
# El mateix per a la part de test:
X_test <- scale(res$test[1:16])
X_test <- array_reshape(X_test, c(nrow(X_test), 16))
X_test <- as.matrix(X_test)
# X_test <- (data.matrix(X_test))
Y_test <- as.numeric(as.matrix(res$test[17]))


dim(X_train)
length(Y_train)
dim(X_test)
length(Y_test)

ylabels<-vector()
ylabels[Y_train=="Positive"]<-1
ylabels[Y_train=="Negative"]<-0


ytestlabels<-vector()
ytestlabels[Y_test=="Positive"]<-1
ytestlabels[Y_test=="Negative"]<-0


model <- keras_model_sequential() 

if (FLAGS$l3 == 0) {
  model %>% 
    layer_dense(units = 10, activation = 'relu', input_shape = c(16)) %>%
    layer_dropout(rate = FLAGS$drop) %>%
    layer_dense(units = 10, activation = 'relu', kernel_regularizer = regularizer_l2(FLAGS$regul)) %>%
    layer_dense(units = 1, activation = 'sigmoid')
  } else if (FLAGS$l3 == 1) {
  model %>% 
    layer_dense(units = 20, activation = 'relu', input_shape = c(16)) %>% 
    layer_dropout(rate = FLAGS$drop) %>%
      layer_dense(units = 10, activation = 'relu', kernel_regularizer = regularizer_l2(FLAGS$regul)) %>%
    layer_dense(units = 5, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'sigmoid')
  }

summary(model)

model %>%compile(
  optimizer = "rmsprop",
  loss = 'binary_crossentropy',
  metric = "acc"
)



# plot(history)

callbacks_list<-list(
  callback_early_stopping(
    monitor = "acc",
    patience=20
  ),
  callback_model_checkpoint(
    filepath = "mymodels.h5",
    monitor = "acc",
    save_best_only = T
  ),
  callback_reduce_lr_on_plateau(
    monitor="acc",
    factor = 0.1,
    patience = 2
  )
)

history <- model %>% fit(
  X_train, Y_train,
  epochs = 30, batch_size = 256,
  validation_split = 0.4,
  callbacks = callbacks_list
)


# x<-as.matrix(X_train)

# model %>% fit(
#   X_train,Y_train, 
#   epochs = 30, batch_size = 256, callbacks = callbacks_list
# )

model %>%
  evaluate(as.matrix(X_test), Y_test)
