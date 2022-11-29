library(corrplot)
library(ggplot2)
#library(mgcv)
library(gam)
library(SuperLearner)
library(dplyr)                      # For the "filter" function
library(splitTools)                 # For the "partition" function
library(caret)
library(performanceEstimation)
library(ggm)
library(MLmetrics)

#monk 1
#cambiare il path per eseguire su altri computer
monk_train_1_hot = read.csv("./Documents/ML_project/monk_1_train_hot.csv",header = FALSE)
#controllo se esistono valori NA
colSums(is.na(monk_train_1_hot))
# creo input - output train
set.seed(1)
split_id <- partition(monk_train_1_hot$V1, p = c(train = 0.7, valid = 0.3))
train_output_1 = monk_train_1_hot[split_id$train,1]
train_input_1 = monk_train_1_hot[split_id$train,-1]
validation_input_1 = monk_train_1_hot[split_id$valid,-1]
validation_output_1 = monk_train_1_hot[split_id$valid,1]
# creo input - output test
monk_test_1_hot = read.csv("./Documents/ML_project/monk_1_test_hot.csv", header = FALSE)
colSums(is.na((monk_test_1_hot)))
test_output_1 = monk_test_1_hot$V1
test_input_1 = subset(monk_test_1_hot,select = -V1)
#provo SuperLearner, se aggiungo SL.knn da errore durante le previsioni
#setto seme per riprodurre la CV ogni volta
#provare a vedere se si può cambiare iperparametri, es KSVM (vedi slide 9 SVM-other-info)
set.seed(3)
tune_ranger = list(num.trees = c(500,1000,2000),
                   mtry = c(floor(sqrt(ncol(train_input_1))),ncol(train_input_1)))
learner_ranger = create.Learner("SL.ranger", tune = tune_ranger, detailed_names = TRUE, name_prefix = "ranger")
tune_svm_rbf = list(kernel = "rbfdot", sigma = c(0.06, 0.01, 0.1), C = c(0.8, 1, 1.2))
learner_svm_rbf = create.Learner("SL.ksvm", tune = tune_svm_rbf, detailed_names = TRUE, name_prefix = "ksvm")
#tune_svm_poly = list(kernel = "polydot", degree = c(2, 2.5, 3), offset = 1, C = c(0.8, 1, 1.2))
#tune_svm_poly = list(kernel = "polydot")
#tune_svm_tanh = list(kernel = "tanhdot")
#learner_svm_poly = create.Learner("SL.ksvm", tune = tune_svm_poly, detailed_names = TRUE, name_prefix = "ksvm")
#learner_svm_tanh = create.Learner("SL.ksvm", tune = tune_svm_tanh, detailed_names = TRUE, name_prefix = "ksvm")
sl1 <- SuperLearner(Y = train_output_1, X = train_input_1, newX = validation_input_1, family = binomial(),
                         SL.library = c("SL.glm",learner_ranger$names ,learner_svm_rbf$names),
                         verbose = TRUE,cvControl=list(10,TRUE) ,control = list(TRUE, TRUE))
sl1

# previsioni VALIDATION SET
pred_rocr = ROCR::prediction(sl1$SL.predict, validation_output_1)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc
#previsioni TEST SET
pred1 = predict(sl1, test_input_1, onlySL = TRUE)
pred_rocr = ROCR::prediction(pred1$pred, test_output_1)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc

### provare a fare risk assessment con CV.Superlearner, forse non serve
# (num_cores = RhpcBLASctl::get_num_cores())
# options(mc.cores = num_cores)
# set.seed(1, "L'Ecuyer-CMRG")
# cv_sl1 = CV.SuperLearner(Y = train_output_1, X = train_input_1, family = binomial(),
#                         cvControl = list(V = 10),parallel = "multicore",saveAll = TRUE,
#                         innerCvControl = list(list(V=10)),SL.library = c("SL.glm", "SL.ranger", "SL.ksvm","SL.knn"),
#                         verbose = TRUE)
# summary(cv_sl1)
# plot(cv_sl1) + theme_bw()
# # Review meta-weights (coefficients) from a CV.SuperLearner object
# review_weights = function(cv_sl) {
#   meta_weights = coef(cv_sl)
#   means = colMeans(meta_weights)
#   sds = apply(meta_weights, MARGIN = 2,  FUN = sd)
#   mins = apply(meta_weights, MARGIN = 2, FUN = min)
#   maxs = apply(meta_weights, MARGIN = 2, FUN = max)
#   # Combine the stats into a single matrix.
#   sl_stats = cbind("mean(weight)" = means, "sd" = sds, "min" = mins, "max" = maxs)
#   # Sort by decreasing mean weight.
#   sl_stats[order(sl_stats[, 1], decreasing = TRUE), ]
# }
# #mostra qual è il miglior modello per ogni fold
# table(simplify2array(cv_sl1$whichDiscreteSL))
# print(review_weights(cv_sl1), digits = 3)

#monk 2
monk_train_2_hot=read.csv("./Documents/ML_project/monk_2_train_hot.csv",header = FALSE)
colSums(is.na(monk_train_2_hot))
set.seed(1)
split_id <- partition(monk_train_2_hot$V1, p = c(train = 0.7, valid = 0.3))
train_output_2 = monk_train_2_hot[split_id$train,1]
train_input_2 = monk_train_2_hot[split_id$train,-1]
validation_input_2 = monk_train_2_hot[split_id$valid,-1]
validation_output_2 = monk_train_2_hot[split_id$valid,1]
monk_test_2_hot = read.csv("./Documents/ML_project/monk_2_test_hot.csv",header = FALSE)
test_output_2 = monk_test_2_hot$V1
test_input_2 = subset(monk_test_2_hot,select = -V1)
set.seed(1)
sl2 <- SuperLearner(Y = train_output_2, X = train_input_2, newX = validation_input_2,family = binomial(),
                    SL.library = c("SL.glm",learner_ranger$names,learner_svm_rbf$names),
                    verbose = TRUE, cvControl=list(10,TRUE),control = list(TRUE, TRUE))
sl2

# previsioni VALIDATION SET
pred_rocr = ROCR::prediction(sl2$SL.predict, validation_output_2)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc
# previsioni TEST SET
pred2 = predict(sl2, test_input_2, onlySL = TRUE)
pred_rocr = ROCR::prediction(pred2$pred, test_output_2)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc

#monk3
monk_train_3_hot=read.csv("./Documents/ML_project/monk_3_train_hot.csv",header = FALSE)
set.seed(1)
split_id <- partition(monk_train_3_hot$V1, p = c(train = 0.7, valid = 0.3))
train_output_3 = monk_train_3_hot[split_id$train,1]
train_input_3 = monk_train_3_hot[split_id$train,-1]
validation_input_3 = monk_train_3_hot[split_id$valid,-1]
validation_output_3 = monk_train_3_hot[split_id$valid,1]
monk_test_3_hot = read.csv("./Documents/ML_project/monk_3_test_hot.csv",header = FALSE)
test_output_3 = monk_test_3_hot$V1
test_input_3 = subset(monk_test_3_hot,select = -V1)
set.seed(3)
sl3 <- SuperLearner(Y = train_output_3, X = train_input_3, newX = validation_input_3,family = binomial(),
                    SL.library = c("SL.glm",learner_ranger$names,learner_svm_rbf$names),
                    verbose = TRUE, cvControl=list(10,TRUE),control = list(TRUE, TRUE))
sl3

#previsioni VALIDATION SET
pred_rocr = ROCR::prediction(sl3$SL.predict, validation_output_3)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc
#previsioni TEST SET
pred3 = predict(sl3, test_input_3, onlySL = TRUE)
pred_rocr = ROCR::prediction(pred3$pred, test_output_3)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc

