# mean euclidean error function
mean_euclidean_error = function(y1_out, y1_target, y2_out, y2_target){
  value = 0
  if (length(y1_out) != length(y1_target) | length(y2_out) != length(y2_target)){
    print("Error in input dimensions")
    return(0)
  }
  for (i in 1:length(y1_out)) {
    y1 = (y1_out[i] - y1_target[i])^2
    y2 = (y2_out[i] - y2_target[i])^2
    root = sqrt(y1 + y2)
    value = value + root
  }
  return(value / length(y1_out))
}


### calcoliamo il MEE del training e della validation per la cross validation del SL
### training MEE
training_mean_euclidean_error_v2 = function(sl_cup_1, sl_cup_2){
  training_mee = 0 # vettore in cui salvo i MEE calcolati in ogni fold
  coef_1 = unname(sl_cup_1$coef) # prendo il vettori dei pesi del SL1
  coef_2 = unname(sl_cup_2$coef) # prendo il vettori dei pesi del SL2
  
  for(j in 1:length(sl_cup_1$cvFitLibrary)) { #stesso valore di sl_cup_2$cvFitLibrary
    # calcolo le predizioni del training set dei modelli con pesi non nulli del SL1
    p1_tot_training = replicate(length(train_output_1_cup[-sl_cup_1$validRows[j][[1]]]), 0)
    for (i in 1:length(sl_cup_1$cvFitLibrary[[j]])) {
      if (coef_1[i] > 0) {
        p1_tot_training = p1_tot_training +
                          predict(sl_cup_1$cvFitLibrary[[j]][[i]], train_input_cup[-sl_cup_1$validRows[j][[1]],],
                                  family = sl_cup_1$family) * coef_1[i]
      }
    }
    
    # calcolo le predizioni del training set dei modelli con pesi non nulli del SL2
    p2_tot_training = replicate(length(train_output_2_cup[-sl_cup_2$validRows[j][[1]]]), 0)
    for (i in 1:length(sl_cup_2$cvFitLibrary[[j]])) {
      if (coef_2[i] > 0) {
        p2_tot_training = p2_tot_training +
                          predict(sl_cup_2$cvFitLibrary[[j]][[i]], train_input_cup[-sl_cup_2$validRows[j][[1]],],
                                  family = sl_cup_2$family) * coef_2[i]
      }
    }
    
    # calcolo MEE per il primo fold e aggiungo il valore al vettore dei MEE
    training_mee = training_mee + mean_euclidean_error(p1_tot_training, train_output_1_cup[-sl_cup_1$validRows[j][[1]]],
                                                       p2_tot_training, train_output_2_cup[-sl_cup_2$validRows[j][[1]]])
  }  
  #restituisco la media del MEE
  return(training_mee / sl_cup_1$cvControl$V)
}


### validation MEE
### stesse operazione del training MEE, cambiano solo le righe.
### in questo caso prendo le righe del validation set per ogni corrispettivo fold
validation_mean_euclidean_error = function(sl_cup_1, sl_cup_2){
  validation_mee = 0
  coef_1 = unname(sl_cup_1$coef)
  coef_2 = unname(sl_cup_2$coef)
  
  for(j in 1:length(sl_cup_1$cvFitLibrary)) { #stesso valore di sl_cup_2$cvFitLibrary
    p1_tot_validation = replicate(length(train_output_1_cup[sl_cup_1$validRows[j][[1]]]), 0)
    for (i in 1:length(sl_cup_1$cvFitLibrary[[j]])) {
      if (coef_1[i] > 0) {
        p1_tot_validation = p1_tot_validation +
          predict(sl_cup_1$cvFitLibrary[[j]][[i]], train_input_cup[sl_cup_1$validRows[j][[1]],],
                  family = sl_cup_1$family) * coef_1[i]
      }
    }
    
    p2_tot_validation = replicate(length(train_output_2_cup[sl_cup_2$validRows[j][[1]]]), 0)
    for (i in 1:length(sl_cup_2$cvFitLibrary[[j]])) {
      if (coef_2[i] > 0) {
        p2_tot_validation = p2_tot_validation +
          predict(sl_cup_2$cvFitLibrary[[j]][[i]], train_input_cup[sl_cup_2$validRows[j][[1]],],
                  family = sl_cup_2$family) * coef_2[i]
      }
    }
    
    validation_mee = validation_mee + mean_euclidean_error(p1_tot_validation, train_output_1_cup[sl_cup_1$validRows[j][[1]]],
                                                           p2_tot_validation, train_output_2_cup[sl_cup_2$validRows[j][[1]]])
  }
  
  return(validation_mee / sl_cup_1$cvControl$V)
}
