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
tune_ranger_cup_assestment = list(num.trees = c(150, 250, 350, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 4000),
                                  mtry = c(2, 3, 4))
learner_ranger_cup_assestment = create.Learner("SL.ranger", tune = tune_ranger_cup_assestment, detailed_names = TRUE,
                                               name_prefix = "ranger")

set.seed(12)
sl_cup_ranger_1 <- SuperLearner(Y = train_output_1_cup, X = train_input_cup,family = gaussian(),
                                SL.library = learner_ranger_cup_assestment$names,
                                verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))

sl_cup_ranger_1
val = data.frame(sl_cup_ranger_1$coef,sl_cup_ranger_1$cvRisk,sl_cup_ranger_1$times$train[3])
write.csv(val,file = "grid_search_results/sl_cup_ranger1_second.csv")
set.seed(12)
sl_cup_ranger_2 <- SuperLearner(Y = train_output_2_cup, X = train_input_cup, family = gaussian(),
                                SL.library = learner_ranger_cup_assestment$names,
                                verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))
sl_cup_ranger_2
val = data.frame(sl_cup_ranger_2$coef,sl_cup_ranger_2$cvRisk,sl_cup_ranger_2$times$train[3])
write.csv(val,file = "grid_search_results/sl_cup_ranger2_second.csv")
# modelli per ksvm
tune_svm_rbf_cup_assestment = list(kernel = "rbfdot",
                                   sigma = c(0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 2.5, 3, 3.5),
                                   C = c(0.1, 0.3, 0.7, 1, 2, 3, 5, 6, 8, 10, 100, 1000))
learner_svm_rbf_cup_assestment = create.Learner("SL.ksvm", tune = tune_svm_rbf_cup_assestment, detailed_names = TRUE,
                                                name_prefix = "ksvm")

set.seed(12)
sl_cup_svm_1 <- SuperLearner(Y = train_output_1_cup, X = train_input_cup,family = gaussian(),
                             SL.library = learner_svm_rbf_cup_assestment$names,
                             verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))
sl_cup_svm_1
val = data.frame(sl_cup_svm_1$coef,sl_cup_svm_1$cvRisk,sl_cup_svm_1$times$train[3])
write.csv(val,file = "grid_search_results/sl_cup_svm1_second.csv")
set.seed(12)
sl_cup_svm_2 <- SuperLearner(Y = train_output_2_cup, X = train_input_cup, family = gaussian(),
                             SL.library = learner_svm_rbf_cup_assestment$names,
                             verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))
sl_cup_svm_2
val = data.frame(sl_cup_svm_2$coef,sl_cup_svm_2$cvRisk,sl_cup_svm_2$times$train[3])
write.csv(val,file = "grid_search_results/sl_cup_svm2_second.csv")
# modelli per ridge e lasso
tune_glmenet_cup_assestment = list(alpha = c(0, 0.1, 0.2, 0.6, 1),
                                   nlambda = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
                                   useMin = TRUE)
learner_glmnet_cup_assestment = create.Learner("SL.glmnet", tune = tune_glmenet_cup_assestment, detailed_names = TRUE,
                                               name_prefix = "glmnet" )
set.seed(12)
sl_cup_glmnet_1 <- SuperLearner(Y = train_output_1_cup, X = train_input_cup,family = gaussian(),
                                SL.library = learner_glmnet_cup_assestment$names,
                                verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))
sl_cup_glmnet_1
val = data.frame(sl_cup_glmnet_1$coef,sl_cup_glmnet_1$cvRisk,sl_cup_glmnet_1$times$train[3])
write.csv(val,file = "grid_search_results/sl_cup_glmnet1_second.csv")
set.seed(12)
sl_cup_glmnet_2 <- SuperLearner(Y = train_output_2_cup, X = train_input_cup, family = gaussian(),
                                SL.library = learner_glmnet_cup_assestment$names,
                                verbose = TRUE, cvControl = list(10, FALSE), control = list(TRUE, TRUE))
sl_cup_glmnet_2
val = data.frame(sl_cup_glmnet_2$coef,sl_cup_glmnet_2$cvRisk,sl_cup_glmnet_2$times$train[3])
write.csv(val,file = "grid_search_results/sl_cup_glmnet2_second.csv")

### previsioni TEST SET, non necessario se passiamo a Superlearner newX
pred_cup_ranger_1 = predict.SuperLearner(object = sl_cup_ranger_1, newdata = test_input_cup, onlySL = TRUE)
pred_cup_ranger_2 = predict.SuperLearner(object = sl_cup_ranger_2, newdata = test_input_cup, onlySL = TRUE)
pred_cup_svm_1 = predict.SuperLearner(object = sl_cup_svm_1, newdata = test_input_cup, onlySL = TRUE)
pred_cup_svm_2 = predict.SuperLearner(object = sl_cup_svm_2, newdata = test_input_cup, onlySL = TRUE)
pred_cup_gmlnet_1 = predict.SuperLearner(object = sl_cup_glmnet_1, newdata = test_input_cup, onlySL = TRUE)
pred_cup_gmlnet_2 = predict.SuperLearner(object = sl_cup_glmnet_2, newdata = test_input_cup, onlySL = TRUE)
###
mae_ranger_1 = MAE(pred_cup_ranger_1$pred, test_output_1_cup)
mae_ranger_2 = MAE(pred_cup_ranger_2$pred, test_output_2_cup)
mae_svm_1 = MAE(pred_cup_svm_1$pred, test_output_1_cup)
mae_svm_2 = MAE(pred_cup_svm_2$pred, test_output_2_cup)
mae_gmlnet_1 = MAE(pred_cup_gmlnet_1$pred, test_output_1_cup)
mae_gmlnet_2 = MAE(pred_cup_gmlnet_2$pred, test_output_2_cup)

accuracy_ranger_1 = 100 - (mae_ranger_1 / abs(mean(test_output_1_cup))*100)
accuracy_ranger_2 = 100 - (mae_ranger_2 / abs(mean(test_output_2_cup))*100)
accuracy_svm_1 = 100 - (mae_svm_1 / abs(mean(test_output_1_cup))*100)
accuracy_svm_2 = 100 - (mae_svm_2 / abs(mean(test_output_2_cup))*100)
accuracy_gmlnet_1 = 100 - (mae_gmlnet_1 / abs(mean(test_output_1_cup))*100)
accuracy_gmlnet_2 = 100 - (mae_gmlnet_2 / abs(mean(test_output_2_cup))*100)

accuracy_ranger_1
accuracy_ranger_2
accuracy_svm_1
accuracy_svm_2
accuracy_gmlnet_1
accuracy_gmlnet_2

# MEE sul training set
training_mean_euclidean_error_v2(sl_cup_ranger_1, sl_cup_ranger_2)
training_mean_euclidean_error_v2(sl_cup_svm_1, sl_cup_svm_2)
training_mean_euclidean_error_v2(sl_cup_glmnet_1, sl_cup_glmnet_2)
# MEE sul validation set
validation_mean_euclidean_error(sl_cup_ranger_1, sl_cup_ranger_2)
validation_mean_euclidean_error(sl_cup_svm_1, sl_cup_svm_2)
validation_mean_euclidean_error(sl_cup_glmnet_1, sl_cup_glmnet_2)
# MEE sul test set
mean_euclidean_error(pred_cup_ranger_1$pred, test_output_1_cup, pred_cup_ranger_2$pred, test_output_2_cup)
mean_euclidean_error(pred_cup_svm_1$pred, test_output_1_cup, pred_cup_svm_2$pred, test_output_2_cup)
mean_euclidean_error(pred_cup_gmlnet_1$pred, test_output_1_cup, pred_cup_gmlnet_2$pred, test_output_2_cup)
