# Working environment: Intel i5-12600k, 32g memory
# R version: 4.1.2
library(data.table)
library(caret)
library(gamlss)
library(pracma)
library(lightgbm)
library(parallel)

#------ Part I. data pre-processing ------
dat <- fread('freMTPL2freq.csv')
dat$ClaimNb <- ifelse(dat$ClaimNb <= 4, dat$ClaimNb, 4)
dat$Area<- as.numeric(as.factor(dat$Area))
dat$VehBrand <- as.numeric(as.factor(dat$VehBrand))
dat$VehGas <- as.numeric(as.factor(dat$VehGas))
dat$Region <-  as.numeric(as.factor(dat$Region))
dat <- cbind(dat[, c(1,2,3,4,9,10,12)], apply(dat[, c(5,6,7,8,11)], 2, scale))

# the seed used to create testing sets in this article, can be changed
set.seed(221)
cv_folds = createFolds(y = dat$ClaimNb, k = 10)
set.seed(NULL)

# the values of lambda used to train M13, can be added to the tuning procedure
M11.lambda.mu <- c(0, 0, 0, 0, 100, 0, 0, 0, 100, 0)
M11.lambda.p <- c(400, 300, 200, 300, 400, 500, 500, 300, 200, 300)

#------ Part II. dual-parameter boosted ZIP trees ------
custom_objective_ZIP_mu <- function(preds, dtrain){
  # grad and hessian for updating mu
  y <- getinfo(dtrain, 'label')
  w <- getinfo(dtrain, 'weight')
  
  mu.pred <- w * exp(preds)
  terms1 <- w * exp(preds - w * exp(preds))
  terms2 <- ZIP.p + (1 - ZIP.p) * exp(-w * exp(preds))
  
  grad <- ifelse(y == 0, 
                 (1-ZIP.p) * terms1 / terms2, # for I(yi = 0)
                 mu.pred - y # for I(yi = 1,2,3,...)
  )
  hess <- ifelse(y == 0, 
                 (1-ZIP.p) / terms2^2 * (terms1 * (1-mu.pred) * terms2 + (1-ZIP.p) * terms1^2), # for I(yi = 0)
                 mu.pred # for I(yi = 1,2,3,...)
  )
  return(list(grad = grad, hess = hess))
}
custom_objective_ZIP_p <- function(preds, dtrain){
  # grad and hessian for updating p
  y <- getinfo(dtrain, 'label')
  
  grad <- ifelse(y == 0, 
                 1/(1 + exp(ZIP.mu+preds)) - 1/(1 + exp(preds)), # for I(yi = 0)
                 exp(preds) / (1 + exp(preds)) # for I(yi = 1,2,3,...)
  )
  hess <- ifelse(y == 0, 
                 exp(preds) / (1 + exp(preds))^2 - exp(ZIP.mu+preds) / (1 + exp(ZIP.mu+preds))^2, # for I(yi = 0)
                 exp(preds) / (1 + exp(preds))^2 # for I(yi = 1,2,3,...)
  )
  return(list(grad = grad, hess = hess))
}

boost_ZIP2_dart <- function(curr_data, curr_params_mu, curr_params_p, max.trees, drop.rate){
  dtrain.mu <- lgb.Dataset(
    data = as.matrix(curr_data[, 4:ncol(curr_data), with = FALSE]), 
    label = curr_data$ClaimNb, 
    weight = curr_data$Exposure, 
    categorical_feature = c("Area","VehBrand","VehGas","Region"), 
    free_raw_data = FALSE
  )
  dtrain.p <- lgb.Dataset(
    data = as.matrix(curr_data[, 4:ncol(curr_data), with = FALSE]), 
    label = curr_data$ClaimNb, 
    weight = curr_data$Exposure, 
    categorical_feature = c("Area","VehBrand","VehGas","Region"), 
    free_raw_data = FALSE
  )
  
  T.trees <- 0
  save.mu.model <- save.p.model <- list()
  save.mu.score <- save.p.score <- matrix(0, nrow = nrow(curr_data), ncol = max.trees)
  while(T.trees < max.trees){
    T.trees <- T.trees + 1
    if(T.trees == 1) ZIP.p <<- rep(0, nrow(curr_data))
    
    ## 1. update mu: when T.trees >= 2, start the dropout procedure
    if(T.trees == 1){
      mu.tree.weights = c(1)
    }else{
      # select the dropped trees and generate the initial values of boosting
      mu.idx <- 1:(T.trees-1)
      if(T.trees == 2) mu.drop.idx <- NULL
      else mu.drop.idx <- mu.idx[rbinom((T.trees-1), 1, drop.rate) == 1]
      mu.select.idx <- mu.idx[!(mu.idx %in% mu.drop.idx)]
      
      if(length(mu.select.idx) == 0){
        mu.score <- rep(0, nrow(curr_data))
      }else{
        mu.weights <- matrix(mu.tree.weights, nrow = nrow(curr_data), ncol = (T.trees-1), byrow = T)
        mu.score <- apply(save.mu.score[, mu.select.idx, drop = F] * mu.weights[, mu.select.idx, drop = F], 1, sum)
      }
      setinfo(dtrain.mu, "init_score", mu.score)
      
      # calculate weights for the new and existing trees
      mu.tree.weights <- c(mu.tree.weights, 1)
      mu.scale.weights <- rep(1, T.trees)
      mu.scale.weights[mu.drop.idx] <- length(mu.drop.idx) / (length(mu.drop.idx) + 1)
      mu.scale.weights[T.trees] <- 1 / (length(mu.drop.idx) + 1)
      mu.tree.weights <- mu.tree.weights * mu.scale.weights
    }
    # train a new base learner
    mu.model <- lgb.train(params = curr_params_mu, data = dtrain.mu, obj = custom_objective_ZIP_mu,  
                          nrounds = 1, verbose = -1)
    save.mu.model[[T.trees]] <- mu.model
    save.mu.score[, T.trees] <- predict(mu.model, as.matrix(curr_data[, 4:ncol(curr_data), with = FALSE]), rawscore = T)
    # update mu for next iteration
    mu.weights <- matrix(mu.tree.weights, nrow = nrow(curr_data), ncol = T.trees, byrow = T)
    mu.train <- apply(save.mu.score[, 1:T.trees, drop = F] * mu.weights[, 1:T.trees, drop = F], 1, sum)
    ZIP.mu <<- curr_data$Exposure * exp(mu.train)
    if(max(ZIP.mu) > 100) stop(paste('Abnormal estimations generated during dropout, retrying...'))
    
    ## 2. update p: when T.trees >= 2, start the dropout procedure
    if(T.trees == 1){
      p.tree.weights = c(1)
    }else{
      # select the dropped trees and generate the initial values of boosting
      p.idx <- 1:(T.trees-1)
      if(T.trees == 2) p.drop.idx <- NULL
      else p.drop.idx <- p.idx[rbinom((T.trees-1), 1, drop.rate) == 1]
      p.select.idx <- p.idx[!(p.idx %in% p.drop.idx)]
      
      if(length(p.select.idx) == 0){
        p.score <- rep(0, nrow(curr_data))
      }else{
        p.weights <- matrix(p.tree.weights, nrow = nrow(curr_data), ncol = (T.trees-1), byrow = T)
        p.score <- apply(save.p.score[, p.select.idx, drop = F] * p.weights[, p.select.idx, drop = F], 1, sum)
      }
      setinfo(dtrain.p, "init_score", p.score)
      
      # calculate weights for the new and existing trees
      p.tree.weights <- c(p.tree.weights, 1)
      p.scale.weights <- rep(1, T.trees)
      p.scale.weights[p.drop.idx] <- length(p.drop.idx) / (length(p.drop.idx) + 1)
      p.scale.weights[T.trees] <- 1 / (length(p.drop.idx) + 1)
      p.tree.weights <- p.tree.weights * p.scale.weights
    }
    # train a new base learner
    p.model <- lgb.train(params = curr_params_p, data = dtrain.p, obj = custom_objective_ZIP_p, 
                         nrounds = 1, verbose = -1)
    save.p.model[[T.trees]] <- p.model
    save.p.score[, T.trees] <- predict(p.model, as.matrix(curr_data[, 4:ncol(curr_data), with = FALSE]), rawscore = T)
    # update p for next iteration
    p.weights <- matrix(p.tree.weights, nrow = nrow(curr_data), ncol = T.trees, byrow = T)
    p.train <- apply(save.p.score[, 1:T.trees, drop = F] * p.weights[, 1:T.trees, drop = F], 1, sum)
    ZIP.p <<- sigmoid(p.train)
    
    if(T.trees %% 50 == 0) print(paste('currently', 2*T.trees, 'updates'))
  }
  
  return(list(mu.model = save.mu.model, mu.tree.weights = mu.tree.weights,
              p.model = save.p.model, p.tree.weights = p.tree.weights))
}
predict_dartscores <- function(new_data, model = NULL, model.weight = NULL){
  max.trees <- length(model)
  
  score.matrix <- matrix(0, nrow = nrow(new_data), ncol = max.trees)
  for(T.trees in 1:max.trees){
    scores <- predict(model[[T.trees]], as.matrix(new_data[, 4:ncol(new_data), with = FALSE]), rawscore = T)
    score.matrix[, T.trees] <- scores * model.weight[T.trees]
  }
  predict.scores <- apply(score.matrix, 1, sum)
  return(predict.scores)
}
ZIP_logloss <- function(y_true, ZIP.mu, ZIP.p){
  loglike <- numeric(length(y_true))
  
  # the ZIP will degenerate to a Poisson or single-point distribution when ZIP.p<0 or ZIP.p>1
  idx1 <- which(ZIP.p <= 1e-3); idx2 <- which(ZIP.p >= 1 - 1e-3); idx3 <- which(ZIP.p > 1e-3 & ZIP.p < 1 - 1e-3)
  if(length(idx1)!=0) loglike[idx1] <- dPO(y_true[idx1], mu = ZIP.mu[idx1], log = TRUE)
  if(length(idx2)!=0) loglike[idx2] <- ifelse(y_true[idx2] == 0, 0, -Inf)
  if(length(idx3)!=0) loglike[idx3] <- dZIP(y_true[idx3], mu = ZIP.mu[idx3], sigma = ZIP.p[idx3], log = TRUE)
  return(-sum(loglike))
}

## In the following example, we fix a random seed and ignore the tuning procedure of dropout.rate; 
## when dropout.rate is not fine-tuned, the (cross-validation) logloss of M13 is about 13730.
## The results shown in the article included the parameter-tuning and did not use the random seed.
set.seed(233)
record.dropout <- numeric(10)
y_list <- ZIP2.p.pred <- ZIP2.mu.pred <- list()
for(k in 1:10){
  train_data <- dat[-cv_folds[[k]], ]; test_data <- dat[cv_folds[[k]], ]
  
  # ### 1. tune parameters using the validation set
  # valid_folds <- as.numeric(createDataPartition(y = train_data$ClaimNb, p = 0.3, list = FALSE))
  # tune_dropout <- c(0.01, 0.02, 0.05, 0.1)
  # 
  # cl = makeCluster(getOption('cl.cores', 4))
  # clusterExport(cl, varlist = c('train_data', 'valid_folds', 'tune_dropout',
  #                               'M11.lambda.mu', 'M11.lambda.p', 'k',
  #                               'custom_objective_ZIP_mu', 'custom_objective_ZIP_p',
  #                               'boost_ZIP2_dart', 'predict_dartscores', 'ZIP_logloss'))
  # tune_results <- parLapply(cl, 1:length(tune_dropout), fun = function(i){
  #   library(gamlss); library(pracma); library(lightgbm)
  # 
  #   curr_params_mu = list(num_leaves = 31, learning_rate = 1,
  #                         lambda_l2 = M11.lambda.mu[k], min_sum_hessian_in_leaf = 200,
  #                         force_row_wise = TRUE, max_bin = 20)
  #   curr_params_p = list(num_leaves = 31, learning_rate = 1,
  #                        lambda_l2 = M11.lambda.p[k], min_sum_hessian_in_leaf = 200,
  #                        force_row_wise = TRUE, max_bin = 20)
  #   repeat{
  #     mod <- try(boost_ZIP2_dart(curr_data = train_data[-valid_folds, ], curr_params_mu, curr_params_p, 
  #                                max.trees = 250, drop.rate = tune_dropout[i]), silent = T)
  #     if(!('try-error' %in% class(mod))) break
  #   }
  # 
  #   mu.valid <- predict_dartscores(new_data = train_data[valid_folds, ], 
  #                                  model = mod$mu.model, model.weight = mod$mu.tree.weights)
  #   ZIP.mu <- train_data[valid_folds, ]$Exposure * exp(mu.valid)
  #   p.valid <- predict_dartscores(new_data = train_data[valid_folds, ], 
  #                                 model = mod$p.model, model.weight = mod$p.tree.weights)
  #   ZIP.p <- sigmoid(p.valid)
  # 
  #   return(ZIP_logloss(y_true = train_data[valid_folds, ]$ClaimNb, ZIP.mu = ZIP.mu, ZIP.p = ZIP.p))
  # })
  # stopCluster(cl); rm(cl)
  # 
  # record.dropout[k] <- tune_dropout[which.min(as.numeric(tune_results))]
  # print(cbind(tune_dropout, as.numeric(tune_results)))
  # print(paste(k, ": ", record.dropout[k], sep = ''))
  # rm(valid_folds, tune_results)
  record.dropout[k] <- 0.05 # using this value for simple illustrations
  
  ### 2. model training
  curr_params_mu = list(num_leaves = 31, learning_rate = 1, 
                        lambda_l2 = M11.lambda.mu[k], min_sum_hessian_in_leaf = 200, 
                        force_row_wise = TRUE, max_bin = 20)
  curr_params_p = list(num_leaves = 31, learning_rate = 1, 
                       lambda_l2 = M11.lambda.p[k], min_sum_hessian_in_leaf = 200, 
                       force_row_wise = TRUE, max_bin = 20)
  repeat{
    mod <- try(boost_ZIP2_dart(curr_data = train_data, curr_params_mu, curr_params_p, 
                               max.trees = 250, drop.rate = record.dropout[k]), silent = T)
    if(!('try-error' %in% class(mod))) break
  }
  rm(ZIP.mu, ZIP.p, curr_params_mu, curr_params_p)
  
  ### 3. model prediction
  ## Once the R program is exited, the predicted values can no longer be calculated by using 'mod'.
  ## If one needs to save the training results of each tree, please use lgb.save() to export them as txt files.
  y_list[[k]] <- test_data$ClaimNb
  mu.pred <- predict_dartscores(new_data = test_data, model = mod$mu.model, model.weight = mod$mu.tree.weights)
  ZIP2.mu.pred[[k]] <- test_data$Exposure * exp(mu.pred)
  p.pred <- predict_dartscores(new_data = test_data, model = mod$p.model, model.weight = mod$p.tree.weights)
  ZIP2.p.pred[[k]] <- sigmoid(p.pred)

  rm(train_data, test_data, mu.pred, p.pred, mod)
  lgb.unloader(wipe = TRUE)
  print(k)
}
mean(apply(array(1:10), 1, function(k){ 
  ZIP_logloss(y_true = y_list[[k]], ZIP.mu = ZIP2.mu.pred[[k]], ZIP.p = ZIP2.p.pred[[k]])
}))
