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
split_id_cup = partition(cup_train$V2, p = c(train = 0.7, test = 0.3))
train_input_cup = cup_train[split_id_cup$train, c(-10, -11)]
train_output_1_cup = cup_train[split_id_cup$train, 10]
train_output_2_cup = cup_train[split_id_cup$train, 11]
test_input_cup = cup_train[split_id_cup$test, c(-10, -11)]
test_output_1_cup = cup_train[split_id_cup$test, 10]
test_output_2_cup = cup_train[split_id_cup$test, 11]


#Superlearner con i modelli che avevano il minor errore nella grid search

# modelli per random forest
tune_ranger_cup_2mtry = list(num.trees = c(250,500, 750, 1000, 1750, 2000, 2500, 3000), mtry = 3)
learner_ranger_cup2mtry = create.Learner("SL.ranger", tune = tune_ranger_cup_2mtry, detailed_names = TRUE,
                                    name_prefix = "ranger")
# modelli per ksvm
tune_svm_rbf_cup = list(kernel = "rbfdot", sigma = c(2.5, 0.001, 0.2, 3.5), C = c(1, 2, 3, 0.7))
learner_svm_rbf_cup = create.Learner("SL.ksvm", tune = tune_svm_rbf_cup, detailed_names = TRUE,
                                     name_prefix = "ksvm")
# modelli per ridge e lasso
tune_glmenet_cup_0.1 = list(alpha = 0.1, nlambda = c(80, 20, 50, 90), useMin = TRUE)
learner_glmnet_cup_0.1 = create.Learner("SL.glmnet", tune = tune_glmenet_cup_0.1, detailed_names = TRUE,
                                    name_prefix = "glmnet" )
tune_glmenet_cup_0.6 = list(alpha = 0.6, nlambda = c(20, 60), useMin = TRUE)
learner_glmnet_cup_0.6 = create.Learner("SL.glmnet", tune = tune_glmenet_cup_0.6, detailed_names = TRUE,
                                        name_prefix = "glmnet" )
tune_glmenet_cup_0.3 = list(alpha = 0.3, nlambda = 100, useMin = TRUE)
learner_glmnet_cup_0.3 = create.Learner("SL.glmnet", tune = tune_glmenet_cup_0.3, detailed_names = TRUE,
                                        name_prefix = "glmnet" )

#input1
set.seed(33)
sl_cup_1_best = SuperLearner(Y = train_output_1_cup, X = train_input_cup,family = gaussian(),
                                   SL.library = c(learner_ranger_cup2mtry$names,
                                                  learner_svm_rbf_cup$names,
                                                  learner_glmnet_cup_0.1$names, learner_glmnet_cup_0.6$names, learner_glmnet_cup_0.3$names),
                                   verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))

sl_cup_1_best
val = data.frame(sl_cup_1_best$coef, sl_cup_1_best$cvRisk, sl_cup_1_best$times$train[3])
write.csv(val,file = "final_SL_results/sl_cup_1_best.csv")
#input2
set.seed(33)
sl_cup_2_best = SuperLearner(Y = train_output_2_cup, X = train_input_cup, family = gaussian(),
                                   SL.library = c(learner_ranger_cup2mtry$names,
                                                  learner_svm_rbf_cup$names,
                                                  learner_glmnet_cup_0.1$names, learner_glmnet_cup_0.6$names, learner_glmnet_cup_0.3$names),
                                   verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))
sl_cup_2_best
val = data.frame(sl_cup_2_best$coef, sl_cup_2_best$cvRisk, sl_cup_2_best$times$train[3])
write.csv(val,file = "final_SL_results/sl_cup_2_best.csv")
### previsioni TEST SET
pred_cup_1 = predict.SuperLearner(object = sl_cup_1_best, newdata = test_input_cup, onlySL = TRUE)
pred_cup_2 = predict.SuperLearner(object = sl_cup_2_best, newdata = test_input_cup, onlySL = TRUE)

###
mae1 = MAE(pred_cup_1$pred, test_output_1_cup)
mae2 = MAE(pred_cup_2$pred, test_output_2_cup)

accuracy1 = 100 - (mae1 / abs(mean(test_output_1_cup)) * 100)
accuracy2 = 100 - (mae2 / abs(mean(test_output_2_cup)) * 100)

accuracy1
accuracy2

# MEE sul training set
training_mean_euclidean_error_v2(sl_cup_1_best, sl_cup_2_best)
# MEE sul validation set
validation_mean_euclidean_error(sl_cup_1_best, sl_cup_2_best)
# MEE sul test set
mean_euclidean_error(pred_cup_1$pred, test_output_1_cup, pred_cup_2$pred, test_output_2_cup)

### Predizioni BLIND-TEST
blind_cup_test = read.csv("./CUP/ML-CUP22-TS_noHeader.csv", header = FALSE)
blind_cup_test = blind_cup_test[,-1]
pred_blind_cup_1 = predict.SuperLearner(object = sl_cup_1_best, newdata = blind_cup_test, onlySL = TRUE)
pred_blind_cup_2 = predict.SuperLearner(object = sl_cup_2_best, newdata = blind_cup_test, onlySL = TRUE)
value = data.frame(pred_blind_cup_1$pred, pred_blind_cup_2$pred)
write.csv(value, file = "./ensemble_team_ML-CUP22.csv")

#a = "# Name1  Surname1	Name2 Surname2\n# Team Name\n# ML-CUP22\n# Submission Date (e.g. 20/11/2022)"
