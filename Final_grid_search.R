library(MLmetrics)
library(SuperLearner)
library(performanceEstimation)
library(splitTools)
library(sjmisc)
source("MEE_functions.R")
#CUP
cup_train = read.csv("./CUP/ML-CUP22-TR_noHeader.csv", header = FALSE)
cup_train = cup_train[c(-1)]
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
tune_ranger_cup = list(num.trees = c(150,250,500, 1000, 2000, 2500), mtry = c(2,3,4))
learner_ranger_cup = create.Learner("SL.ranger", tune = tune_ranger_cup, detailed_names = TRUE,
                                    name_prefix = "ranger")
tune_ranger_350_9_cup = list(num.trees = 350, mtry = ncol(train_input_cup))
learner_ranger_350_9_cup = create.Learner("SL.ranger", tune = tune_ranger_350_9_cup, detailed_names = TRUE,
                                          name_prefix = "ranger")
# modelli per ksvm
tune_svm_rbf_cup = list(kernel = "rbfdot", sigma = c(2.5, 0.01,0.2,0.5), C = c(0.1, 0.7,5,10,1000))
learner_svm_rbf_cup = create.Learner("SL.ksvm", tune = tune_svm_rbf_cup, detailed_names = TRUE,
                                     name_prefix = "ksvm")
# modelli per ridge e lasso
tune_glmenet_cup = list(alpha = c(0,0.1,0.6),nlambda = c(10,20,50), useMin = TRUE)
learner_glmnet_cup = create.Learner("SL.glmnet", tune = tune_glmenet_cup, detailed_names = TRUE,
                                    name_prefix = "glmnet" )


#Superlearner con meno modelli

# modelli per random forest
tune_ranger_cup_1 = list(num.trees = 2500, mtry = 3)
tune_ranger_cup_2 = list(num.trees = 2000, mtry = 2)
tune_ranger_cup_3 = list(num.trees = 1000, mtry = 2)
tune_ranger_cup_4 = list(num.trees = 150, mtry = 3)
tune_ranger_cup_5 = list(num.trees = 250, mtry = 2)
tune_ranger_cup_6 = list(num.trees = 500, mtry = 4)

learner_ranger_cup_lim_1 = create.Learner("SL.ranger", tune = tune_ranger_cup_1, detailed_names = TRUE,
                                          name_prefix = "ranger")
learner_ranger_cup_lim_2 = create.Learner("SL.ranger", tune = tune_ranger_cup_2, detailed_names = TRUE,
                                          name_prefix = "ranger")
learner_ranger_cup_lim_3 = create.Learner("SL.ranger", tune = tune_ranger_cup_3, detailed_names = TRUE,
                                          name_prefix = "ranger")
learner_ranger_cup_lim_4 = create.Learner("SL.ranger", tune = tune_ranger_cup_4, detailed_names = TRUE,
                                          name_prefix = "ranger")
learner_ranger_cup_lim_5 = create.Learner("SL.ranger", tune = tune_ranger_cup_5, detailed_names = TRUE,
                                          name_prefix = "ranger")
learner_ranger_cup_lim_6 = create.Learner("SL.ranger", tune = tune_ranger_cup_6, detailed_names = TRUE,
                                          name_prefix = "ranger")
# modelli per ksvm
tune_svm_rbf_cup_1 = list(kernel = "rbfdot", sigma = 2.5, C = 0.7)
tune_svm_rbf_cup_2 = list(kernel = "rbfdot", sigma = 0.01, C = 5)
tune_svm_rbf_cup_3 = list(kernel = "rbfdot", sigma = 2.5, C = 10)
tune_svm_rbf_cup_4 = list(kernel = "rbfdot", sigma = 0.5, C = 0.1)

learner_svm_rbf_cup_lim_1 = create.Learner("SL.ksvm", tune = tune_svm_rbf_cup_1, detailed_names = TRUE,
                                     name_prefix = "ksvm")
learner_svm_rbf_cup_lim_2 = create.Learner("SL.ksvm", tune = tune_svm_rbf_cup_2, detailed_names = TRUE,
                                           name_prefix = "ksvm")
learner_svm_rbf_cup_lim_3 = create.Learner("SL.ksvm", tune = tune_svm_rbf_cup_3, detailed_names = TRUE,
                                           name_prefix = "ksvm")
learner_svm_rbf_cup_lim_4 = create.Learner("SL.ksvm", tune = tune_svm_rbf_cup_4, detailed_names = TRUE,
                                           name_prefix = "ksvm")
# modelli per ridge e lasso
tune_glmenet_cup_1 = list(alpha = 0.1, nlambda = 50, useMin = TRUE)
tune_glmenet_cup_2 = list(alpha = 0.1, nlambda = 20, useMin = TRUE)
tune_glmenet_cup_3 = list(alpha = 0.6, nlambda = 20, useMin = TRUE)

learner_glmnet_cup_lim_1 = create.Learner("SL.glmnet", tune = tune_glmenet_cup_1, detailed_names = TRUE,
                                    name_prefix = "glmnet" )
learner_glmnet_cup_lim_2 = create.Learner("SL.glmnet", tune = tune_glmenet_cup_2, detailed_names = TRUE,
                                          name_prefix = "glmnet" )

learner_glmnet_cup_lim_3 = create.Learner("SL.glmnet", tune = tune_glmenet_cup_3, detailed_names = TRUE,
                                          name_prefix = "glmnet" )


#input1
set.seed(12)
sl_cup_1_final_lim <- SuperLearner(Y = train_output_1_cup, X = train_input_cup,family = gaussian(),
                                   SL.library = c(learner_ranger_cup_lim_1$names, learner_ranger_cup_lim_2$names, learner_ranger_cup_lim_3$names, learner_ranger_cup_lim_4$names, learner_ranger_cup_lim_5$names, learner_ranger_cup_lim_6$names,
                                              learner_svm_rbf_cup_lim_1$names, learner_svm_rbf_cup_lim_2$names, learner_svm_rbf_cup_lim_3$names, learner_svm_rbf_cup_lim_4$names,
                                              learner_glmnet_cup_lim_1$names, learner_glmnet_cup_lim_2$names, learner_glmnet_cup_lim_3$names),
                                   verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))

sl_cup_1_final_lim
#input2
set.seed(12)
sl_cup_2_final_lim <- SuperLearner(Y = train_output_2_cup, X = train_input_cup, family = gaussian(),
                                   SL.library = c(learner_ranger_cup_lim_1$names, learner_ranger_cup_lim_2$names, learner_ranger_cup_lim_3$names, learner_ranger_cup_lim_4$names, learner_ranger_cup_lim_5$names, learner_ranger_cup_lim_6$names,
                                                  learner_svm_rbf_cup_lim_1$names, learner_svm_rbf_cup_lim_2$names, learner_svm_rbf_cup_lim_3$names, learner_svm_rbf_cup_lim_4$names,
                                                  learner_glmnet_cup_lim_1$names, learner_glmnet_cup_lim_2$names, learner_glmnet_cup_lim_3$names),
                                   verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))
sl_cup_2_final_lim


#IN ALTERNATIVA...

# modelli per random forest
tune_ranger_cup = list(num.trees = seq(250, 5000, by=50), mtry = 1:ncol(train_input_cup))
learner_ranger_cup = create.Learner("SL.ranger", tune = tune_ranger_cup, detailed_names = TRUE,
                                    name_prefix = "ranger")
# modelli per ksvm
tune_svm_rbf_cup = list(kernel = "rbfdot", sigma = seq(0.001, 10, by=0.01), C = seq(0.001, 10, by=0.01))
learner_svm_rbf_cup = create.Learner("SL.ksvm", tune = tune_svm_rbf_cup, detailed_names = TRUE,
                                     name_prefix = "ksvm")
# modelli per ridge e lasso
tune_glmenet_cup = list(alpha = seq(0, 1, by=0.1), nlambda = seq(100, 1000, by=50), useMin = c(TRUE,FALSE))
learner_glmnet_cup = create.Learner("SL.glmnet", tune = tune_glmenet_cup, detailed_names = TRUE,
                                    name_prefix = "glmnet" )



#input1
set.seed(12)
sl_cup_1_final <- SuperLearner(Y = train_output_1_cup, X = train_input_cup,family = gaussian(),
                         SL.library = c(learner_ranger_cup$names, learner_ranger_350_9_cup$names ,learner_svm_rbf_cup$names, learner_glmnet_cup$names),
                         verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))

sl_cup_1_final
#input2
set.seed(12)
sl_cup_2_final <- SuperLearner(Y = train_output_2_cup, X = train_input_cup, family = gaussian(),
                         SL.library = c(learner_ranger_cup$names, learner_ranger_350_9_cup$names, learner_svm_rbf_cup$names, learner_glmnet_cup$names),
                         verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))
sl_cup_2_final


### previsioni TEST SET, non necessario se passiamo a Superlearner newX
pred_cup_1_final = predict.SuperLearner(object = sl_cup_1_final, newdata = test_input_cup, onlySL = TRUE)
pred_cup_2_final = predict.SuperLearner(object = sl_cup_2_final, newdata = test_input_cup, onlySL = TRUE)

pred_cup_1_final = predict.SuperLearner(object = sl_cup_1_final_lim, newdata = test_input_cup, onlySL = TRUE)
pred_cup_2_final = predict.SuperLearner(object = sl_cup_2_final_lim, newdata = test_input_cup, onlySL = TRUE)
###
mae1_final = MAE(pred_cup_1_final$pred, test_output_1_cup)
mae2_final = MAE(pred_cup_2_final$pred, test_output_2_cup)

accuracy1_final = 100 - (mae1_final / abs(mean(test_output_1_cup))*100)
accuracy2_final = 100 - (mae2_final / abs(mean(test_output_2_cup))*100)

accuracy1_final
accuracy2_final
# MSE sul test
mse1_final = MSE(pred_cup_1_final$pred, test_output_1_cup)
mse1_final
mse2_final = MSE(pred_cup_2_final$pred, test_output_2_cup)
mse2_final
# MEE sul training set
training_mean_euclidean_error_v2(sl_cup_1_final, sl_cup_2_final)
training_mean_euclidean_error_v2(sl_cup_1_final_lim, sl_cup_2_final_lim)
# MEE sul validation set
validation_mean_euclidean_error(sl_cup_1_final, sl_cup_2_final)
validation_mean_euclidean_error(sl_cup_1_final_lim, sl_cup_2_final_lim)
# MEE sul test set
mean_euclidean_error(pred_cup_1_final$pred, test_output_1_cup, pred_cup_2_final$pred, test_output_2_cup)
