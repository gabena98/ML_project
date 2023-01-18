#library(corrplot)
#library(ggplot2)
library(SuperLearner)                     # For the "filter" function
library(splitTools)                 # For the "partition" function
library(caret)
library(performanceEstimation)
library(ggm)
library(MLmetrics)

#monk 1
monk_train_1_hot = read.csv("./MONK/monk_1_train_hot.csv", header = FALSE)
#controllo se esistono valori NA
colSums(is.na(monk_train_1_hot))
# creo input - output train
set.seed(1)
train_input_1 = monk_train_1_hot[,-1]
train_output_1 = monk_train_1_hot[,1]
# creo input - output test
monk_test_1_hot = read.csv("./MONK/monk_1_test_hot.csv", header = FALSE)
colSums(is.na((monk_test_1_hot)))
test_output_1 = monk_test_1_hot[,1]
test_input_1 = monk_test_1_hot[,-1]
#setto seme per riprodurre la CV ogni volta
set.seed(3)

tune_ranger = list(num.trees = c(500, 750, 1000, 1500, 1750, 2000),
                   mtry = c(floor(sqrt(ncol(train_input_1))),ncol(train_input_1)))
learner_ranger = create.Learner("SL.ranger", tune = tune_ranger, detailed_names = TRUE, name_prefix = "ranger")

tune_svm_rbf = list(kernel = "rbfdot", sigma = c(0.06, 0.2, 0.01, 0.1), C = c(0.8, 1, 1.2))
learner_svm_rbf = create.Learner("SL.ksvm", tune = tune_svm_rbf, detailed_names = TRUE, name_prefix = "ksvm")

sl1 <- SuperLearner(Y = train_output_1, X = train_input_1, family = binomial(),
                         SL.library = c("SL.glm",learner_ranger$names ,learner_svm_rbf$names),
                         verbose = TRUE,cvControl=list(10,TRUE) ,control = list(TRUE, TRUE))
sl1

# AUC TRAINING SET
pred <- ifelse(sl1$SL.predict< 0.5, 0, 1)
confusionMatrix(as.factor(pred), as.factor(train_output_1), mode = "everything", positive = "1")
#MSE TRAINING SET
pred_rocr = ROCR::prediction(sl1$SL.predict, train_output_1)
rmse = ROCR::performance(pred_rocr, measure = "rmse", x.measure = "cutoff")@y.values[[1]]
mse=rmse^2
mse
# AUC TEST SET
pred1 = predict(sl1, test_input_1, onlySL = TRUE)
pred <- ifelse(pred1$pred< 0.5, 0, 1)
confusionMatrix(as.factor(pred), as.factor(test_output_1), mode = "everything", positive = "1")
#MSE TEST SET
pred_rocr = ROCR::prediction(pred1$pred, test_output_1)
rmse = ROCR::performance(pred_rocr, measure = "rmse", x.measure = "cutoff")@y.values[[1]]
mse=rmse^2
mse

#monk 2
monk_train_2_hot=read.csv("./MONK/monk_2_train_hot.csv",header = FALSE)
colSums(is.na(monk_train_2_hot))
set.seed(1)
train_input_2 = monk_train_2_hot[,-1]
train_output_2 = monk_train_2_hot[,1]
monk_test_2_hot = read.csv("./MONK/monk_2_test_hot.csv",header = FALSE)
test_output_2 = monk_test_2_hot[,1]
test_input_2 = monk_test_2_hot[,-1]
set.seed(1)
sl2 <- SuperLearner(Y = train_output_2, X = train_input_2,family = binomial(),
                    SL.library = c("SL.glm",learner_ranger$names,learner_svm_rbf$names),
                    verbose = TRUE, cvControl=list(10,TRUE),control = list(TRUE, TRUE))
sl2

# AUC TRAINING SET
pred <- ifelse(sl2$SL.predict< 0.5, 0, 1)
confusionMatrix(as.factor(pred), as.factor(train_output_2), mode = "everything", positive = "1")
#MSE TRAINING SET
pred_rocr = ROCR::prediction(sl2$SL.predict, train_output_2)
rmse = ROCR::performance(pred_rocr, measure = "rmse", x.measure = "cutoff")@y.values[[1]]
mse=rmse^2
mse
# AUC TEST SET
pred2 = predict(sl2, test_input_2, onlySL = TRUE)
pred <- ifelse(pred2$pred< 0.5, 0, 1)
confusionMatrix(as.factor(pred), as.factor(test_output_2), mode = "everything", positive = "1")
#MSE TEST SET
pred_rocr = ROCR::prediction(pred2$pred, test_output_2)
rmse = ROCR::performance(pred_rocr, measure = "rmse", x.measure = "cutoff")@y.values[[1]]
mse=rmse^2
mse

#monk3
monk_train_3_hot=read.csv("./MONK/monk_3_train_hot.csv",header = FALSE)
set.seed(1)
train_input_3 = monk_train_3_hot[,-1]
train_output_3 = monk_train_3_hot[,1]
monk_test_3_hot = read.csv("./MONK/monk_3_test_hot.csv",header = FALSE)
test_output_3 = monk_test_3_hot[,1]
test_input_3 = monk_test_3_hot[,-1]
set.seed(3)
sl3 <- SuperLearner(Y = train_output_3, X = train_input_3,family = binomial(),
                    SL.library = c("SL.glm",learner_ranger$names,learner_svm_rbf$names),
                    verbose = TRUE, cvControl=list(10,TRUE),control = list(TRUE, TRUE))
sl3

#AUC TRAINING SET
pred <- ifelse(sl3$SL.predict< 0.5, 0, 1)
confusionMatrix(as.factor(pred), as.factor(train_output_3), mode = "everything", positive = "1")
#MSE TRAINING SET
pred_rocr = ROCR::prediction(sl3$SL.predict, train_output_3)
rmse = ROCR::performance(pred_rocr, measure = "rmse", x.measure = "cutoff")@y.values[[1]]
mse=rmse^2
mse
#AUC TEST SET
pred3 = predict(sl3, test_input_3, onlySL = TRUE)
pred <- ifelse(pred3$pred< 0.5, 0, 1)
confusionMatrix(as.factor(pred), as.factor(test_output_3), mode = "everything", positive = "1")
#MSE TRAINING SET
pred_rocr = ROCR::prediction(pred3$pred, test_output_3)
rmse = ROCR::performance(pred_rocr, measure = "rmse", x.measure = "cutoff")@y.values[[1]]
mse=rmse^2
mse