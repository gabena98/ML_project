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


#Superlearner con i modelli scelti negli ensemble dell'ultima grid search

# modelli per random forest
tune_ranger_cup = list(num.trees = c(150, 250, 350, 500, 1000, 2000, 2500), mtry = c(2, 3, 4, ncol(train_input_cup)))
learner_ranger_cup = create.Learner("SL.ranger", tune = tune_ranger_cup, detailed_names = TRUE,
                                    name_prefix = "ranger")
                                    
# modelli per ksvm
tune_svm_rbf_cup = list(kernel = "rbfdot", sigma = c(2.5, 0.01,0.2,0.5), C = c(0.1, 0.7,5,10,1000))
learner_svm_rbf_cup = create.Learner("SL.ksvm", tune = tune_svm_rbf_cup, detailed_names = TRUE,
                                     name_prefix = "ksvm")
# modelli per ridge e lasso
tune_glmenet_cup = list(alpha = c(0,0.1,0.6),nlambda = c(10,20,50), useMin = TRUE)
learner_glmnet_cup = create.Learner("SL.glmnet", tune = tune_glmenet_cup, detailed_names = TRUE,
                                    name_prefix = "glmnet" )

#input1
set.seed(33)
sl_cup_1_final = SuperLearner(Y = train_output_1_cup, X = train_input_cup,family = gaussian(),
                         SL.library = c(learner_ranger_cup$names, learner_svm_rbf_cup$names, learner_glmnet_cup$names),
                         verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))

sl_cup_1_final
#input2
set.seed(33)
sl_cup_2_final = SuperLearner(Y = train_output_2_cup, X = train_input_cup, family = gaussian(),
                         SL.library = c(learner_ranger_cup$names, learner_svm_rbf_cup$names, learner_glmnet_cup$names),
                         verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))
sl_cup_2_final


### previsioni TEST SET, non necessario se passiamo a Superlearner newX
pred_cup_1_final = predict.SuperLearner(object = sl_cup_1_final, newdata = test_input_cup, onlySL = TRUE)
pred_cup_2_final = predict.SuperLearner(object = sl_cup_2_final, newdata = test_input_cup, onlySL = TRUE)

###
mae1_final = MAE(pred_cup_1_final$pred, test_output_1_cup)
mae2_final = MAE(pred_cup_2_final$pred, test_output_2_cup)

accuracy1_final = 100 - (mae1_final / abs(mean(test_output_1_cup)) * 100)
accuracy2_final = 100 - (mae2_final / abs(mean(test_output_2_cup)) * 100)

accuracy1_final
accuracy2_final

# MSE sul test
mse1_final = MSE(pred_cup_1_final$pred, test_output_1_cup)
mse1_final
mse2_final = MSE(pred_cup_2_final$pred, test_output_2_cup)
mse2_final

# MEE sul training set
training_mean_euclidean_error_v2(sl_cup_1_final, sl_cup_2_final)
# MEE sul validation set
validation_mean_euclidean_error(sl_cup_1_final, sl_cup_2_final)
# MEE sul test set
mean_euclidean_error(pred_cup_1_final$pred, test_output_1_cup, pred_cup_2_final$pred, test_output_2_cup)