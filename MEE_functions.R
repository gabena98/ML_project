# mean euclidean error function
mean_euclidean_error = function(y1_out, y1_target, y2_out, y2_target){
  value = 0
  if (length(y1_out)!=length(y1_target) | length(y2_out)!=length(y2_target)){
    print("Error in input dimensions")
    return(0)
  }
  for (i in 1:length(y1_out)) {
    y1 = (y1_out[i] - y1_target[i])^2
    y2 = (y2_out[i] - y2_target[i])^2
    root = sqrt(y1+y2)
    value = value+root
  }
  return(value/length(y1_out))
}
### calcoliamo il MEE del training e della validation per la cross validation del SL
### le funzioni non sono parametrizzabili  :*-(
### training MEE
training_mean_euclidean_error = function(sl_cup_1,sl_cup_2){
  training_mee = 0 # vettore in cui salvo i MEE calcolati in ogni fold
  #1 primo fold
  # calcolo predizioni primo SL
  coef_1 = unname(sl_cup_1$coef) # prendo il vettori dei pesi del SL1
  #name_1 = names(sl_cup_1$coef) # salvo il nome dei modelli, non lo uso
  #name_1 = name_1[coef_1!=0] # salvo il nome dei modelli con peso non nullo, non lo uso
  coef_1 = coef_1[coef_1!=0] # salvo i pesi dei modelli non nulli
  # calcolo le predizioni del training set dei modelli con pesi non nulli del SL1
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`1`$ranger_1000_3_All, train_input_cup[-sl_cup_1$validRows$`1`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`1`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[-sl_cup_1$validRows$`1`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`1`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_1$validRows$`1`,], family= sl_cup_1$family)
  # calcolo predizioni SL1
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  coef_2 = unname(sl_cup_2$coef) # prendo il vettori dei pesi del SL2
  #name_2 = names(sl_cup_2$coef)
  #name_2 = name_2[coef_2!=0]
  coef_2 = coef_2[coef_2!=0] # salvo i pesi dei modelli non nulli
  # calcolo le predizioni del training set dei modelli con pesi non nulli del SL2
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`1`$ranger_2000_3_All, train_input_cup[-sl_cup_2$validRows$`1`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`1`$ranger_500_9_All, train_input_cup[-sl_cup_2$validRows$`1`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`1`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[-sl_cup_2$validRows$`1`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`1`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_2$validRows$`1`,], family= sl_cup_2$family)
  # calcolo predizioni SL2
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  # calcolo MEE per il primo fold e aggiungo il valore al vettore dei MEE
  training_mee = training_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[-sl_cup_1$validRows$`1`],p2_tot_training,train_output_2_cup[-sl_cup_2$validRows$`1`])
  #2 secondo fold
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`2`$ranger_1000_3_All, train_input_cup[-sl_cup_1$validRows$`2`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`2`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[-sl_cup_1$validRows$`2`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`2`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_1$validRows$`2`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`2`$ranger_2000_3_All, train_input_cup[-sl_cup_2$validRows$`2`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`2`$ranger_500_9_All, train_input_cup[-sl_cup_2$validRows$`2`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`2`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[-sl_cup_2$validRows$`2`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`2`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_2$validRows$`2`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  training_mee = training_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[-sl_cup_1$validRows$`2`],p2_tot_training,train_output_2_cup[-sl_cup_2$validRows$`2`])
  #3
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`3`$ranger_1000_3_All, train_input_cup[-sl_cup_1$validRows$`3`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`3`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[-sl_cup_1$validRows$`3`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`3`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_1$validRows$`3`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`3`$ranger_2000_3_All, train_input_cup[-sl_cup_2$validRows$`3`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`3`$ranger_500_9_All, train_input_cup[-sl_cup_2$validRows$`3`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`3`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[-sl_cup_2$validRows$`3`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`3`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_2$validRows$`3`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  training_mee = training_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[-sl_cup_1$validRows$`3`],p2_tot_training,train_output_2_cup[-sl_cup_2$validRows$`3`])
  #4
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`4`$ranger_1000_3_All, train_input_cup[-sl_cup_1$validRows$`4`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`4`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[-sl_cup_1$validRows$`4`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`4`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_1$validRows$`4`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`4`$ranger_2000_3_All, train_input_cup[-sl_cup_2$validRows$`4`,], family = sl_cup_2$family) # nolint
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`4`$ranger_500_9_All, train_input_cup[-sl_cup_2$validRows$`4`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`4`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[-sl_cup_2$validRows$`4`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`4`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_2$validRows$`4`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4] # nolint
  training_mee = training_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[-sl_cup_1$validRows$`4`],p2_tot_training,train_output_2_cup[-sl_cup_2$validRows$`4`])
  #5
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`5`$ranger_1000_3_All, train_input_cup[-sl_cup_1$validRows$`5`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`5`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[-sl_cup_1$validRows$`5`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`5`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_1$validRows$`5`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`5`$ranger_2000_3_All, train_input_cup[-sl_cup_2$validRows$`5`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`5`$ranger_500_9_All, train_input_cup[-sl_cup_2$validRows$`5`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`5`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[-sl_cup_2$validRows$`5`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`5`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_2$validRows$`5`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  training_mee = training_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[-sl_cup_1$validRows$`5`],p2_tot_training,train_output_2_cup[-sl_cup_2$validRows$`5`])
  #6
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`6`$ranger_1000_3_All, train_input_cup[-sl_cup_1$validRows$`6`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`6`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[-sl_cup_1$validRows$`6`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`6`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_1$validRows$`6`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`6`$ranger_2000_3_All, train_input_cup[-sl_cup_2$validRows$`6`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`6`$ranger_500_9_All, train_input_cup[-sl_cup_2$validRows$`6`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`6`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[-sl_cup_2$validRows$`6`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`6`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_2$validRows$`6`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  training_mee = training_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[-sl_cup_1$validRows$`6`],p2_tot_training,train_output_2_cup[-sl_cup_2$validRows$`6`])
  #7
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`7`$ranger_1000_3_All, train_input_cup[-sl_cup_1$validRows$`7`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`7`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[-sl_cup_1$validRows$`7`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`7`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_1$validRows$`7`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`7`$ranger_2000_3_All, train_input_cup[-sl_cup_2$validRows$`7`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`7`$ranger_500_9_All, train_input_cup[-sl_cup_2$validRows$`7`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`7`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[-sl_cup_2$validRows$`7`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`7`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_2$validRows$`7`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  training_mee = training_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[-sl_cup_1$validRows$`7`],p2_tot_training,train_output_2_cup[-sl_cup_2$validRows$`7`])
  #8
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`8`$ranger_1000_3_All, train_input_cup[-sl_cup_1$validRows$`8`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`8`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[-sl_cup_1$validRows$`8`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`8`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_1$validRows$`8`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`8`$ranger_2000_3_All, train_input_cup[-sl_cup_2$validRows$`8`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`8`$ranger_500_9_All, train_input_cup[-sl_cup_2$validRows$`8`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`8`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[-sl_cup_2$validRows$`8`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`8`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_2$validRows$`8`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  training_mee = training_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[-sl_cup_1$validRows$`8`],p2_tot_training,train_output_2_cup[-sl_cup_2$validRows$`8`])
  #9
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`9`$ranger_1000_3_All, train_input_cup[-sl_cup_1$validRows$`9`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`9`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[-sl_cup_1$validRows$`9`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`9`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_1$validRows$`9`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`9`$ranger_2000_3_All, train_input_cup[-sl_cup_2$validRows$`9`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`9`$ranger_500_9_All, train_input_cup[-sl_cup_2$validRows$`9`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`9`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[-sl_cup_2$validRows$`9`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`9`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_2$validRows$`9`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  training_mee = training_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[-sl_cup_1$validRows$`9`],p2_tot_training,train_output_2_cup[-sl_cup_2$validRows$`9`])
  #10
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`10`$ranger_1000_3_All, train_input_cup[-sl_cup_1$validRows$`10`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`10`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[-sl_cup_1$validRows$`10`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`10`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_1$validRows$`10`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`10`$ranger_2000_3_All, train_input_cup[-sl_cup_2$validRows$`10`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`10`$ranger_500_9_All, train_input_cup[-sl_cup_2$validRows$`10`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`10`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[-sl_cup_2$validRows$`10`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`10`$glmnet_1_100_TRUE_All, train_input_cup[-sl_cup_2$validRows$`10`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  training_mee = training_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[-sl_cup_1$validRows$`10`],p2_tot_training,train_output_2_cup[-sl_cup_2$validRows$`10`])
  #restituisco la media del MEE
  return(training_mee/sl_cup_1$cvControl$V)
}
### validation MEE
### stesse operazione del training MEE, cambiano solo le righe.
### in questo caso prendo le righe del validation set per ogni corrispettivo fold
validation_mean_euclidean_error = function(sl_cup_1,sl_cup_2){
  validation_mee=0
  #1
  # calcolo predizioni primo SL
  coef_1 = unname(sl_cup_1$coef)
  #name_1 = names(sl_cup_2$coef)
  #name_1 = name_1[coef_1!=0]
  coef_1 = coef_1[coef_1!=0]
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`1`$ranger_1000_3_All, train_input_cup[sl_cup_1$validRows$`1`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`1`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[sl_cup_1$validRows$`1`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`1`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_1$validRows$`1`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  coef_2 = unname(sl_cup_2$coef)
  #name_2 = names(sl_cup_2$coef)
  #name_2 = name_2[coef_2!=0]
  coef_2 = coef_2[coef_2!=0]
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`1`$ranger_2000_3_All, train_input_cup[sl_cup_2$validRows$`1`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`1`$ranger_500_9_All, train_input_cup[sl_cup_2$validRows$`1`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`1`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[sl_cup_2$validRows$`1`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`1`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_2$validRows$`1`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  validation_mee = validation_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[sl_cup_1$validRows$`1`],p2_tot_training,train_output_2_cup[sl_cup_2$validRows$`1`])
  #2
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`2`$ranger_1000_3_All, train_input_cup[sl_cup_1$validRows$`2`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`2`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[sl_cup_1$validRows$`2`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`2`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_1$validRows$`2`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`2`$ranger_2000_3_All, train_input_cup[sl_cup_2$validRows$`2`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`2`$ranger_500_9_All, train_input_cup[sl_cup_2$validRows$`2`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`2`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[sl_cup_2$validRows$`2`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`2`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_2$validRows$`2`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  validation_mee = validation_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[sl_cup_1$validRows$`2`],p2_tot_training,train_output_2_cup[sl_cup_2$validRows$`2`])
  #3
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`3`$ranger_1000_3_All, train_input_cup[sl_cup_1$validRows$`3`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`3`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[sl_cup_1$validRows$`3`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`3`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_1$validRows$`3`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`3`$ranger_2000_3_All, train_input_cup[sl_cup_2$validRows$`3`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`3`$ranger_500_9_All, train_input_cup[sl_cup_2$validRows$`3`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`3`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[sl_cup_2$validRows$`3`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`3`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_2$validRows$`3`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  validation_mee = validation_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[sl_cup_1$validRows$`3`],p2_tot_training,train_output_2_cup[sl_cup_2$validRows$`3`])
  #4
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`4`$ranger_1000_3_All, train_input_cup[sl_cup_1$validRows$`4`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`4`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[sl_cup_1$validRows$`4`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`4`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_1$validRows$`4`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`4`$ranger_2000_3_All, train_input_cup[sl_cup_2$validRows$`4`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`4`$ranger_500_9_All, train_input_cup[sl_cup_2$validRows$`4`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`4`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[sl_cup_2$validRows$`4`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`4`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_2$validRows$`4`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  validation_mee = validation_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[sl_cup_1$validRows$`4`],p2_tot_training,train_output_2_cup[sl_cup_2$validRows$`4`])
  #5
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`5`$ranger_1000_3_All, train_input_cup[sl_cup_1$validRows$`5`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`5`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[sl_cup_1$validRows$`5`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`5`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_1$validRows$`5`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`5`$ranger_2000_3_All, train_input_cup[sl_cup_2$validRows$`5`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`5`$ranger_500_9_All, train_input_cup[sl_cup_2$validRows$`5`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`5`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[sl_cup_2$validRows$`5`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`5`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_2$validRows$`5`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  validation_mee = validation_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[sl_cup_1$validRows$`5`],p2_tot_training,train_output_2_cup[sl_cup_2$validRows$`5`])
  #6
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`6`$ranger_1000_3_All, train_input_cup[sl_cup_1$validRows$`6`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`6`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[sl_cup_1$validRows$`6`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`6`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_1$validRows$`6`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`6`$ranger_2000_3_All, train_input_cup[sl_cup_2$validRows$`6`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`6`$ranger_500_9_All, train_input_cup[sl_cup_2$validRows$`6`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`6`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[sl_cup_2$validRows$`6`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`6`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_2$validRows$`6`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  validation_mee = validation_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[sl_cup_1$validRows$`6`],p2_tot_training,train_output_2_cup[sl_cup_2$validRows$`6`])
  #7
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`7`$ranger_1000_3_All, train_input_cup[sl_cup_1$validRows$`7`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`7`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[sl_cup_1$validRows$`7`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`7`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_1$validRows$`7`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`7`$ranger_2000_3_All, train_input_cup[sl_cup_2$validRows$`7`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`7`$ranger_500_9_All, train_input_cup[sl_cup_2$validRows$`7`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`7`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[sl_cup_2$validRows$`7`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`7`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_2$validRows$`7`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  validation_mee = validation_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[sl_cup_1$validRows$`7`],p2_tot_training,train_output_2_cup[sl_cup_2$validRows$`7`])
  #8
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`8`$ranger_1000_3_All, train_input_cup[sl_cup_1$validRows$`8`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`8`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[sl_cup_1$validRows$`8`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`8`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_1$validRows$`8`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`8`$ranger_2000_3_All, train_input_cup[sl_cup_2$validRows$`8`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`8`$ranger_500_9_All, train_input_cup[sl_cup_2$validRows$`8`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`8`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[sl_cup_2$validRows$`8`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`8`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_2$validRows$`8`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  validation_mee = validation_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[sl_cup_1$validRows$`8`],p2_tot_training,train_output_2_cup[sl_cup_2$validRows$`8`])
  #9
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`9`$ranger_1000_3_All, train_input_cup[sl_cup_1$validRows$`9`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`9`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[sl_cup_1$validRows$`9`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`9`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_1$validRows$`9`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`9`$ranger_2000_3_All, train_input_cup[sl_cup_2$validRows$`9`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`9`$ranger_500_9_All, train_input_cup[sl_cup_2$validRows$`9`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`9`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[sl_cup_2$validRows$`9`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`9`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_2$validRows$`9`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  validation_mee = validation_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[sl_cup_1$validRows$`9`],p2_tot_training,train_output_2_cup[sl_cup_2$validRows$`9`])
  #10
  # calcolo predizioni primo SL
  p1_1_training <- predict(sl_cup_1$cvFitLibrary$`10`$ranger_1000_3_All, train_input_cup[sl_cup_1$validRows$`10`,], family = sl_cup_1$family)
  p1_2_training <- predict(sl_cup_1$cvFitLibrary$`10`$ksvm_rbfdot_0.01_1.2_All, train_input_cup[sl_cup_1$validRows$`10`,],family = sl_cup_1$family)
  p1_3_training <- predict(sl_cup_1$cvFitLibrary$`10`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_1$validRows$`10`,], family= sl_cup_1$family)
  p1_tot_training = p1_1_training*coef_1[1]+p1_2_training*coef_1[2]+p1_3_training*coef_1[3]
  #calcolo predizioni secondo SL
  p2_1_training <- predict(sl_cup_2$cvFitLibrary$`10`$ranger_2000_3_All, train_input_cup[sl_cup_2$validRows$`10`,], family = sl_cup_2$family)
  p2_2_training <- predict(sl_cup_2$cvFitLibrary$`10`$ranger_500_9_All, train_input_cup[sl_cup_2$validRows$`10`,],family = sl_cup_2$family)
  p2_3_training <- predict(sl_cup_2$cvFitLibrary$`10`$ksvm_rbfdot_0.1_1.2_All, train_input_cup[sl_cup_2$validRows$`10`,], family= sl_cup_2$family)
  p2_4_training <- predict(sl_cup_2$cvFitLibrary$`10`$glmnet_1_100_TRUE_All, train_input_cup[sl_cup_2$validRows$`10`,], family= sl_cup_2$family)
  p2_tot_training = p2_1_training*coef_2[1]+p2_2_training*coef_2[2]+p2_3_training*coef_2[3]+p2_4_training*coef_2[4]
  validation_mee = validation_mee + mean_euclidean_error(p1_tot_training,train_output_1_cup[sl_cup_1$validRows$`10`],p2_tot_training,train_output_2_cup[sl_cup_2$validRows$`10`])
  
  return(validation_mee/sl_cup_1$cvControl$V)
}
