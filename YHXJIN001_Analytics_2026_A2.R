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
Xu = cbind(data$AveSpeed, data$Sex)
Xv = cbind(data$Carbs, data$Fluid)

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

neural_net = function(Xu, Xv, m, theta, nu, Y=NULL){
  # output: Y_hat: Raw probability output of the logistic function
  # output: pred: prediction value
  # output: E: evaluation error as required
  
  pu = dim(Xu)[2]
  pv = dim(Xv)[2]
  N = dim(Xu)[1]
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
  E=NULL
  if (!is.null(Y)){
    
    base_error = obj(Y, Y_hat) / nrow(Y)
    penalty = (nu / nrow(Y)) * (sum(W1^2) + sum(W2^2) + sum(W3^2))
    E = base_error + penalty
  }
  
  
  return(list(Y_hat = Y_hat, pred = pred, E=E))
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

nu_grid = exp(seq(-8,-0,length
            =25))
# Do grid search on values of nu_grid
errors = rep(1, length(nu_grid))
theta_store = matrix(NA, length(nu_grid), nparams)
for (i in 1:length(nu_grid)){
  nu = nu_grid[i]
  
  obj_train = function(theta_init) {
    result = neural_net(Xtrain_u, Xtrain_v, 4, theta_init, nu, Ytrain)
    return(result$E)
  }
  cat("Training network for nu =", nu, "\n")
  trained = nlm(obj_train, theta, iterlim=1000)
  theta_store[i, ] =trained$estimate
  valid_result = neural_net(Xvalid_u, Xvalid_v, 4, trained$estimate, nu)

  cat("Finished validating nu =", nu, "\n")
  errors[i] = obj(Yvalid, valid_result$Y_hat) / nrow(Xvalid_u)
}
plot(errors ~ log(nu_grid), pch=1, col="firebrick4", type="b")
best_index = which.min(errors)
best_nu = nu_grid[best_index]
theta_best_nu = theta_store[best_index,]

# Response Curve over Carbs, Sex, Fluid

library(colorspace)
color.gradient = function(x, colors=c('coral','purple','skyblue1'), colsteps=25)
{
  colpal = colorRampPalette(colors)
  return( colpal(colsteps)[ findInterval(x, seq(min(x),max(x), length=colsteps)) ] )
}


M=100
carbs_seq = seq(min(data$Carbs),max(data$Carbs),length.out= M)
carbs_mean = mean(data$Carbs)
fluid_seq = seq(min(data$Fluid),max(data$Fluid),length.out= M)
fluid_mean = mean(data$Fluid)




xxv1 = rep(carbs_seq, M)
xxv2 = rep(fluid_seq, each =M)
XXv = cbind(xxv1, xxv2)

xxu1 = rep(12, M^2)
xxu2_male = rep(1, each = M^2)
xxu2_female = rep(0, each=M^2)
XXumale = cbind(xxu1, xxu2_male)
XXufemale = cbind(xxu1, xxu2_female)

plot(xxv2 ~ xxv1, col=color.gradient(response_female$Y_hat), pch=16, cex=0.8, 
     xlab="Carbohydrate intake (gram,)", ylab="Fluid intake (litre)", 
     main="Female: probability of hitting the wall.")
legend("topright", title="Risk of Bombing",
       legend=c("High (Near 100%)", "Medium (~50%)", "Low (Near 0%)"), 
       col=c("skyblue1", "purple", "coral"), pch=16, bg="white", cex=0.5)

plot(xxv2 ~ xxv1, col=color.gradient(response_male$Y_hat), pch=16, cex=0.8, 
     xlab="Carbohydrate intake (gram)", ylab="Fluid intake (litre)", 
     main="Male: probability of hitting the wall.")
legend("topright", title="Risk of Bombing",
       legend=c("High (Near 100%)", "Medium (~50%)", "Low (Near 0%)"), 
       col=c("skyblue1", "purple", "coral"), pch=16, bg="white", cex=0.5)


Xu_male = data.frame(AveSpeed=12, Sex=0)
Xu_male = Xu_male[rep(seq_len(nrow(Xu_male)), each = 100), ]
Xu_female = data.frame(AveSpeed=12, Sex=1)
Xu_female = Xu_female[rep(seq_len(nrow(Xu_female)), each = 100), ]
Xv_carbs = data.frame(Carbs=carbs_seq, fluid=fluid_mean)
Xv_fluid = data.frame(Carbs=carbs_mean, fluid=fluid_seq)
X_carbs = data.frame(AveSpeed=12, Sex=0, Fluid=1)

pred_carbsMale = neural_net(Xu_male, Xv_carbs, 4, theta_best_nu, best_nu)
pred_carbsFemale = neural_net(Xu_female, Xv_carbs, 4, theta_best_nu, best_nu)
pred_fluidMale = neural_net(Xu_male, Xv_fluid, 4, theta_best_nu, best_nu)
pred_fluidFemale =neural_net(Xu_female, Xv_fluid, 4, theta_best_nu, best_nu)


plot(as.vector(pred_carbsMale$Y_hat) ~ carbs_seq, type="l", col="coral1", lwd=2, 
     xlab="Carbs", ylab="Probability of Bombing", main="Response to Carbs by Sex", ylim=c(0,1))
lines(as.vector(pred_carbsFemale$Y_hat) ~ carbs_seq, type="l", col="skyblue4", lwd=2)
legend("bottomleft", legend=c("Male (Sex=0)", "Female (Sex=1)"), 
       col=c("coral1", "skyblue4"), lty=1, lwd=2, cex=0.5)

plot(as.vector(pred_fluidMale$Y_hat) ~ fluid_seq, type="l", col="hotpink", lwd=2, 
     xlab="Fluid", ylab="Probability of Bombing", main="Response to Fluid by Sex")
lines(as.vector(pred_fluidFemale$Y_hat) ~ fluid_seq, type="l", col="black", lwd=2)
legend("bottomleft", legend=c("Male (Sex=0)", "Female (Sex=1)"), 
       col=c("hotpink", "black"), lty=1, lwd=2, cex=0.5)


# Interpretation: It seems like males, in general, are able to consume less carbs, and drink more fluid to avoid getting bombed.
# On the other hand, female atheletes cannot drink as much fluid as males, and require more carbs to not getting bombed

# We recommend male runners to aim to consume between 1l to 1.8l of fluid,, 50g to 90g of carbs when running at 12km/hr
# Female runners should aim to consume between 1l to 1.6l of fluid, 60g to 90g of carbs when running at 12km/hr.

# Empirical advantage of using a split brain network

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
      result = neural_net(Xu_train, Xv_train, 4, theta_init, nu)
      return(result$pred)
    }
    result_fold = nlm(neural_net, theta)
    total_valid_error = neural_net(Xu_valid, Xv_valid, 4, result_fold$esimate, Y_valid, nu)
    
    return (total_valid_error/folds)
  }
  
  
}

# 80-20 splits mean we do 5 folds cross validation
result = nlm(obj_train, theta)




