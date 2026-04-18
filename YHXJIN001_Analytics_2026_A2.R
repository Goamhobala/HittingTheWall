rm(list=ls())
library(Matrix)
# EDA
data = as.data.frame(read.table("HittingTheWall_2026.txt", header=TRUE))
head(data)
summary(data)
# First we check for correlations
pairs(Bombed ~ . , data=data, pch=16, col="hotpink")

# Then we check for distribution of the data
par(mfrow=c(2,2))
hist(data$Carbs)
hist(data$Fluid)
hist(data$AveSpeed)
boxplot(data)
# From the summary and the boxplot, we see that the features have ranges that vary greatly.
# This is bad for neural networks. We must thus standardise the data to make sure no variables
# Dominate
data_scaled = as.data.frame(scale(data))
summary(data_scaled)
par(mfrow = c(1, 1))

softmax = function(Z_l){
  # This fucntion takes in a q x N matrix, with z vector for each observation as columns
  # Outputs a q x N probability matrix
  
  # We first calculatethe max value for each column
  z_max = apply(Z_l, 2, max)
  # turn the max matrix into a diagonal matrix
  Z_max = diag(z_max)
  # Apply exp to both Z_max and Z_l and then calculate exp(z_l - z_max) product
  expZ_max = -exp(Z_max)
  expZ_l = exp(Z_l)
  expDiff = expZ_l %*% expZ_max
  
  # Get the denominators for softmax
  Denominator = diag(1/colSums(expDiff))
  
  S = expDiff %*% Denominator
  return(S)
}
Xu = cbind(data_scaled$AveSpeed, data_scaled$Sex)
Xv = cbind(data_scaled$Carbs, data_scaled$Fluid)

# We should not scale the response variable
Y = cbind(data$Bombed)

sig = function(A){
  return(tanh(A))
}
sig_out = function(a){
  return (1/(1+exp(-a)))
}

obj = function(Y, Y_hat){
  epsilon = 1e-8 # to prevent log(0)
  Y_hat = pmax(pmin(Y_hat, 1-epsilon),epsilon)
  return (sum(-(Y * log(t(Y_hat)) + (1-Y) * log(t(1-Y_hat)))))
}

neural_net = function(Xu, Xv, m, theta, Y, nu){
  pu = dim(Xu)[2]
  pv = dim(Xv)[2]
  N = dim(Y)[1]
  X = cbind(Xu, Xv)
  # 2m part
  
  index = 1:(m*pu)
  W1u = matrix(theta[index], pu, m)

  index = max(index) + 1:(pv*m)
  W1v = matrix(theta[index], pv, m)

  W1 = as.matrix(bdiag(W1u, W1v))

  # 2 layer
  index = max(index)+1:(2*m*2) 
  W2 = matrix(theta[index], 2 * m, 2)

  
  index = max(index)+1:2
  W3 = matrix(theta[index], 2, 1)

  index = max(index) + 1:(m)
  b1u = matrix(theta[index], m, 1)
  index = max(index) + 1:(m)
  b1v = matrix(theta[index], m, 1)
  b1 = rbind(b1u, b1v)
  
  index = max(index) + 1:2
  b2 = matrix(theta[index], 2, 1)
  index = max(index) + 1:1
  b3 = matrix(theta[index], 1, 1)
  
  ones = matrix(1, N, 1)
  A0 = as.matrix(t(X))
  A1 = sig(t(W1) %*% A0 + b1 %*% t(ones))
  A2 = sig(t(W2) %*% A1 + b2 %*% t(ones))
  z_out = t(W3) %*% A2 + b3 %*% t(ones)
  # Since this is a classification problem, we use logistic equation
  Y_hat = sig_out(z_out)
  pred = ifelse(Y_hat >= 0.5, 1, 0)
  E = obj(Y, Y_hat)/N + (nu /N) * (sum(W1^2) + sum(W2^2) + sum(W3^2))
  
  return(list(A1=A1, A2=A2, E=E))
}

set.seed(2026)
m = 4
N = dim(Y)[1]
train_indices = sample(1:N, size=round(0.8*N))

Xtrain_u = Xu[train_indices,]
Xtrain_v = Xv[train_indices,]
Ytrain = as.matrix(Y[train_indices])

Xvalid_u = Xu[-train_indices,]
Xvalid_v = Xv[-train_indices,]
Yvalid = as.matrix(Y[-train_indices])

pu = dim(Xu)[2]
pv = dim(Xv)[2]
nparams = (pu * m) + (pv * m) + 2 * m  + 2 * m * 2 + 2 + 2 + 1
theta = runif(nparams, -1, 1)

nu_grid = exp(seq(-7,-0,length
            =20))
# Do grid search on values of nu_grid
errors = rep(1, length(nu_grid))
for (i in 1:length(nu_grid)){
  nu = nu_grid[i]
  
  obj_train = function(theta_init) {
    result = neural_net(Xtrain_u, Xtrain_v, 4, theta_init, Ytrain, nu)
    return(result$E)
  }
  cat("Training network for nu =", nu, "\n")
  trained = nlm(obj_train,theta)
  valid_result = neural_net(Xvalid_u, Xvalid_v, 4, trained$estimate, Yvalid, nu)

  cat("Finished validating nu =", nu, "\n")
  errors[i] = valid_result$E  
}
lines(errors ~ log(nu_grid), pch=1, col="firebrick4")
best_index = which.min(errors)
best_nu = nu_grid[best_index]



# Just in case we need cross validation
set.seed(2026)
cross_validation = function(Xu, Xv, folds, nu){
  total_valid_error = 0
  N = dim(X)[1]
  fold_size = floor(N/5)
  for (i in 1:folds){
    if (i != folds){
      valid_indices = (i-1) * fold_size : i * fold_size
    }
    else{
      # Handle last trailing values
      valid_indices= (i-1) * fold_size:-1
    }
    Xu_train = Xu[-valid_indices, ]
    Xu_valid = Xu[valid_indices,]
    Xv_train = Xv[-valid_indices, ]
    Xv_valid = Xv[valid_indices,]
    Y_train = as.matrix(Y[-valid_indices])
    Y_valid = as.matrix(Y[valid_indices])
    
    nparams = (pu * m) + (pv * m) + 2 * m  + 2 * m * 2 + 2 + 2 + 1
    theta = runif(nparams, -1, 1)
    
    # Wrapper function to train NN with nlm
    obj_train = function(theta_init) {
      result = neural_net(Xu_train, Xv_train, 4, theta_init, Y_train, nu)
      return(result$E)
    }
    result_fold = nlm(neural_net, theta)
    total_valid_error = neural_net(Xu_valid, Xv_valid, 4, result_fold$esimate, Y_valid, nu)
    
    return (total_valid_error/folds)
  }
  
  
}

# 80-20 splits mean we do 5 folds cross validation
result = nlm(obj_train, theta)




