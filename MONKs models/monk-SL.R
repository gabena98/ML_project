#library(corrplot)
#library(ggplot2)
library(gam)
library(SuperLearner)                 
library(splitTools)                 # For the "partition" function
library(caret)
library(performanceEstimation)
library(ggm)
library(MLmetrics)

#monk 1
#cambiare il path per eseguire su altri computer
monk_train_1=read.table("./MONK/monks-1.train", header = FALSE, sep = "", dec = ".")
#controllo se esistono valori NA
colSums(is.na(monk_train_1))
# creo input - output train
set.seed(1)
train_output_1 = monk_train_1[,1]
train_input_1 = monk_train_1[,c(-1,-8)]
# creo input - output test
monk_test_1 = read.table("./MONK/monks-1.test", header = FALSE, sep = "", dec = ".")
colSums(is.na((monk_test_1)))
test_output_1 = monk_test_1$V1
test_input_1 = subset(monk_test_1,select = c(-V1,-V8))
#setto seme per riprodurre la CV ogni volta
set.seed(3)
tune_ranger = list(num.trees = c(500,1000,2000),
                   mtry = c(floor(sqrt(ncol(train_input_1))),ncol(train_input_1)))
learner_ranger = create.Learner("SL.ranger", tune = tune_ranger, detailed_names = TRUE, name_prefix = "ranger")
tune_svm_rbf = list(kernel = "rbfdot", sigma = c(0.06, 0.01, 0.1), C = c(0.8, 1, 1.2))
learner_svm_rbf = create.Learner("SL.ksvm", tune = tune_svm_rbf, detailed_names = TRUE, name_prefix = "ksvm")

sl1 <- SuperLearner(Y = train_output_1, X = train_input_1, family = binomial(),
                         SL.library = c("SL.glm",learner_ranger$names ,learner_svm_rbf$names),
                         verbose = TRUE,cvControl=list(10,TRUE) ,control = list(TRUE, TRUE))
sl1

# AUC TRAINING SET
pred_rocr = ROCR::prediction(sl1$SL.predict, train_output_1)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc
#previsioni TEST SET
pred1 = predict(sl1, test_input_1, onlySL = TRUE)
pred_rocr = ROCR::prediction(pred1$pred, test_output_1)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc

#monk 2
monk_train_2=read.table("./MONK/monks-2.train",header = FALSE, sep = "", dec = ".")
set.seed(1)
train_output_2 = monk_train_2[,1]
train_input_2 = monk_train_2[,c(-1,-8)]
monk_test_2 = read.table("./MONK/monks-2.test", header = FALSE, sep = "", dec = ".")
test_output_2 = monk_test_2$V1
test_input_2= subset(monk_test_2,select = c(-V1,-V8))
set.seed(3)
sl2 <- SuperLearner(Y = train_output_2, X = train_input_2,family = binomial(),
                    SL.library = c("SL.glm",learner_ranger$names,learner_svm_rbf$names),
                    verbose = TRUE, cvControl=list(10,TRUE),control = list(TRUE, TRUE))
sl2

# AUC TRAINING SET
pred_rocr = ROCR::prediction(sl1$SL.predict, train_output_1)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc
# previsioni TEST SET
pred2 = predict(sl2, test_input_2, onlySL = TRUE)
pred_rocr = ROCR::prediction(pred2$pred, test_output_2)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc

#monk3
monk_train_3=read.table("./MONK/monks-3.train",header = FALSE, sep = "", dec = ".")
set.seed(1)
train_output_3 = monk_train_3[split_id$train,1]
train_input_3 = monk_train_3[split_id$train,c(-1,-8)]
monk_test_3 = read.table("./MONK/monks-3.test", header = FALSE, sep = "", dec = ".")
test_output_3 = monk_test_3$V1
test_input_3 = subset(monk_test_3,select = c(-V1,-V8))
set.seed(3)
sl3 <- SuperLearner(Y = train_output_3, X = train_input_3,family = binomial(),
                    SL.library = c("SL.glm",learner_ranger$names,learner_svm_rbf$names),
                    verbose = TRUE, cvControl=list(10,TRUE),control = list(TRUE, TRUE))
sl3

# AUC TRAINING SET
pred_rocr = ROCR::prediction(sl1$SL.predict, train_output_1)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc
#previsioni TEST SET
pred3 = predict(sl3, test_input_3, onlySL = TRUE)
pred_rocr = ROCR::prediction(pred3$pred, test_output_3)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc