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
monk_train_1=read.table("/Users/gabrielebenanti/Documents/MONK/monks-1.train", header = FALSE, sep = "", dec = ".")
#controllo se esistono valori NA
colSums(is.na(monk_train_1))
# creo input - output train
set.seed(1)
split_id <- partition(monk_train_1$V1, p = c(train = 0.7, valid = 0.3))
train_output_1 = monk_train_1[split_id$train,1]
train_input_1 = monk_train_1[split_id$train,c(-1,-8)]
validation_input_1 = monk_train_1[split_id$valid,c(-1,-8)]
validation_output_1 = monk_train_1[split_id$valid,1]
# creo input - output test
monk_test_1 = read.table("/Users/gabrielebenanti/Documents/MONK/monks-1.test", header = FALSE, sep = "", dec = ".")
colSums(is.na((monk_test_1)))
test_output_1 = monk_test_1$V1
test_input_1 = subset(monk_test_1,select = c(-V1,-V8))
#provo SuperLearner, se aggiungo SL.knn da errore durante le previsioni
#setto seme per riprodurre la CV ogni volta
#provare a vedere se si può cambiare iperparametri, es KSVM (vedi slide 9 SVM-other-info)
set.seed(3)
sl1 <- SuperLearner(Y = train_output_1, X = train_input_1, newX = validation_input_1, family = binomial(),
                         SL.library = c("SL.glm","SL.ranger","SL.ksvm"),
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
(num_cores = RhpcBLASctl::get_num_cores())
options(mc.cores = num_cores)
set.seed(1, "L'Ecuyer-CMRG")
cv_sl1 = CV.SuperLearner(Y = train_output_1, X = train_input_1, family = binomial(),
                        cvControl = list(V = 10),parallel = "multicore",saveAll = TRUE,
                        innerCvControl = list(list(V=10)),SL.library = c("SL.glm", "SL.ranger", "SL.ksvm","SL.knn"),
                        verbose = TRUE)
summary(cv_sl1)
plot(cv_sl1) + theme_bw()
# Review meta-weights (coefficients) from a CV.SuperLearner object
review_weights = function(cv_sl) {
  meta_weights = coef(cv_sl)
  means = colMeans(meta_weights)
  sds = apply(meta_weights, MARGIN = 2,  FUN = sd)
  mins = apply(meta_weights, MARGIN = 2, FUN = min)
  maxs = apply(meta_weights, MARGIN = 2, FUN = max)
  # Combine the stats into a single matrix.
  sl_stats = cbind("mean(weight)" = means, "sd" = sds, "min" = mins, "max" = maxs)
  # Sort by decreasing mean weight.
  sl_stats[order(sl_stats[, 1], decreasing = TRUE), ]
}
#mostra qual è il miglior modello per ogni fold
table(simplify2array(cv_sl1$whichDiscreteSL))
print(review_weights(cv_sl1), digits = 3)

#monk 2
monk_train_2=read.table("/Users/gabrielebenanti/Documents/MONK/monks-2.train",header = FALSE, sep = "", dec = ".")
set.seed(1)
split_id <- partition(monk_train_2$V1, p = c(train = 0.7, valid = 0.3))
train_output_2 = monk_train_2[split_id$train,1]
train_input_2 = monk_train_2[split_id$train,c(-1,-8)]
validation_input_2 = monk_train_2[split_id$valid,c(-1,-8)]
validation_output_2 = monk_train_2[split_id$valid,1]
monk_test_2 = read.table("/Users/gabrielebenanti/Documents/MONK/monks-2.test", header = FALSE, sep = "", dec = ".")
test_output_2 = monk_test_2$V1
test_input_2= subset(monk_test_2,select = c(-V1,-V8))
set.seed(3)
sl2 <- SuperLearner(Y = train_output_2, X = train_input_2, newX = validation_input_2,family = binomial(),
                    SL.library = c("SL.glm","SL.ranger","SL.ksvm"),
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
monk_train_3=read.table("/Users/gabrielebenanti/Documents/MONK/monks-3.train",header = FALSE, sep = "", dec = ".")
set.seed(1)
split_id <- partition(monk_train_3$V1, p = c(train = 0.7, valid = 0.3))
train_output_3 = monk_train_3[split_id$train,1]
train_input_3 = monk_train_3[split_id$train,c(-1,-8)]
validation_input_3 = monk_train_3[split_id$valid,c(-1,-8)]
validation_output_3 = monk_train_3[split_id$valid,1]
monk_test_3 = read.table("/Users/gabrielebenanti/Documents/MONK/monks-3.test", header = FALSE, sep = "", dec = ".")
test_output_3 = monk_test_3$V1
test_input_3 = subset(monk_test_3,select = c(-V1,-V8))
set.seed(3)
sl3 <- SuperLearner(Y = train_output_3, X = train_input_3, newX = validation_input_3,family = binomial(),
                    SL.library = c("SL.glm","SL.ranger","SL.ksvm"),
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

#CUP dataset
ml_cup_tr =read.table("/Users/gabrielebenanti/Library/CloudStorage/OneDrive-UniversityofPisa/Università/magistrale/ML/ML-22-PRJ lecture  package-20221108/ML-CUP22-TR.csv",header = FALSE, sep ="," , dec = ".")
colSums(is.na((ml_cup_tr)))
set.seed(10)
split_id <- partition(ml_cup_tr$V12, p = c(train = 0.7, valid = 0.3))

ml_cup_tr_output=ml_cup_tr[split_id$train,c(11,12)]
ml_cup_tr_input=ml_cup_tr[split_id$train,2:10]

ml_cup_ts_output=ml_cup_tr[split_id$valid,c(11,12)]
ml_cup_ts_input=ml_cup_tr[split_id$valid,2:10]

cup1<- SuperLearner(Y = ml_cup_tr_output$V11, X = ml_cup_tr_input, family = gaussian(),
                    SL.library = c("SL.ksvm"))
cup1
