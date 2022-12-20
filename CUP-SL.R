library(MLmetrics)
library(SuperLearner)
library(performanceEstimation)
library(splitTools)
library(sjmisc)

#CUP
cup_train = read.csv("./CUP/ML-CUP22-TR_noHeader.csv", header = FALSE)
cup_train = cup_train[c(-1)]
{# dataset normalizzato (varianza = 1)
#cup_train_std = std(cup_train, append = FALSE) #questa funzione imposta media = 0, varianza = 1 (standardizzazione)
# oppure
# data_size = c() # vettore che contiene le sd di ogni colonna
# for (i in 1:dim(cup_train)[2]) {
#   data_size=append(data_size,sd(cup_train[,i]))
# }
# cup_train_std = sweep(cup_train,2,data_size,`/`) #dataset normalizzato
#controllo se esistono valori NA
}
colSums(is.na(cup_train))
# creo input - output train
set.seed(1)
# potremmo direttamente dividere in train, test. La valiation la crea dentro la CV
split_id_cup <- partition(cup_train$V2, p = c(train = 0.7, test = 0.3))
train_input_cup = cup_train[split_id_cup$train, c(-10, -11)]
train_output_1_cup = cup_train[split_id_cup$train, 10]
train_output_2_cup = cup_train[split_id_cup$train, 11]
test_input_cup = cup_train[split_id_cup$test, c(-10, -11)]
test_output_1_cup = cup_train[split_id_cup$test,10]
test_output_2_cup = cup_train[split_id_cup$test,11]
#Superlearner
# modelli per random forest
tune_ranger_cup = list(num.trees = c(500,1000,2000), mtry = c(floor(sqrt(ncol(train_input_cup))),
                                                              ncol(train_input_cup)))
learner_ranger_cup = create.Learner("SL.ranger", tune = tune_ranger_cup, detailed_names = TRUE,
                                    name_prefix = "ranger")
# modelli per ksvm
tune_svm_rbf_cup = list(kernel = "rbfdot", sigma = c(0.06, 0.01, 0.1), C = c(0.8, 1, 1.2))
learner_svm_rbf_cup = create.Learner("SL.ksvm", tune = tune_svm_rbf_cup, detailed_names = TRUE,
                                     name_prefix = "ksvm")
# modelli per ridge e lasso
tune_glmenet_cup = list(alpha = c(0,1),nlambda = c(100,1000,5000), useMin = c(TRUE,FALSE))
learner_glmnet_cup = create.Learner("SL.glmnet", tune = tune_glmenet_cup, detailed_names = TRUE,
                                    name_prefix = "glmnet" )

#input1
set.seed(12)
sl_cup_1 <- SuperLearner(Y = train_output_1_cup, X = train_input_cup,family = gaussian(),
                    SL.library = c(learner_ranger_cup$names ,learner_svm_rbf_cup$names, learner_glmnet_cup$names),
                    verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))

sl_cup_1
set.seed(12)
sl_cup_2 <- SuperLearner(Y = train_output_2_cup, X = train_input_cup, family = gaussian(),
                         SL.library = c(learner_ranger_cup$names ,learner_svm_rbf_cup$names, learner_glmnet_cup$names),
                         verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))
sl_cup_2
### previsioni TEST SET, non necessario se passiamo a Superlearner newX
pred_cup_1 = predict.SuperLearner(object = sl_cup_1, newdata = test_input_cup, onlySL = TRUE)
pred_cup_2 = predict.SuperLearner(object = sl_cup_2, newdata = test_input_cup, onlySL = TRUE)
###
mae1 = MAE(pred_cup_1$pred, test_output_1_cup)
mae2 = MAE(pred_cup_2$pred, test_output_2_cup)

accuracy1 = 100 - (mae1 / abs(mean(test_output_1_cup))*100)
accuracy2 = 100 - (mae2 / abs(mean(test_output_2_cup))*100)

accuracy1
accuracy2
# MSE sul test
mse1 = MSE(pred_cup_1$pred, test_output_1_cup)
mse1
mse2 = MSE(pred_cup_2$pred, test_output_2_cup)
mse2
# MEE sul training set
training_mean_euclidean_error(sl_cup_1,sl_cup_2)
# MEE sul validation set
validation_mean_euclidean_error(sl_cup_1,sl_cup_2)
# MEE sul test set
mean_euclidean_error(pred_cup_1$pred,test_output_1_cup,pred_cup_2$pred,test_output_2_cup)
# possiamo calcolarci il MSE del training set una volta scelti i pesi 
# e rifittati i modelli su tutto il training set
mse1_1 = MSE(sl_cup_1$SL.predict, train_output_1_cup)
mse1_1
mse2_1 = MSE(sl_cup_2$SL.predict, train_output_2_cup)
mse2_1
{
# SOLO nel caso di dataset normalizzato
# scalo a varianza originale le variabili predette
# pred1 = sweep(sl_cup_1$SL.predict,2,data_size[10],`*`)
# pred2 = sweep(sl_cup_2$SL.predict,2,data_size[11],`*`)
# validation_output_1_cup_real = cup_train[split_id_cup$test,10]
# validation_output_2_cup_real = cup_train[split_id_cup$test,11]
# mean_euclidean_error(pred1,validation_output_1_cup_real,pred2,validation_output_2_cup_real)
# mse1.1 = MSE(pred1, validation_output_1_cup_real)
# mse1.1
# mse2.1 = MSE(pred2, validation_output_2_cup_real)
# mse2.1
}