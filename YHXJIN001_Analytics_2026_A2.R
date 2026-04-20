rm(list=ls())
library(Matrix)
# EDA
data = as.data.frame(read.table("HittingTheWall_2026.txt", header=TRUE))
head(data)
summary(data)
# First we check for correlations

# 1. Add extra blank space to the bottom (the 3) and top (the 2)
par(oma = c(3, 0, 2, 0)) 

# 2. Draw your pairs plot
pairs(Bombed ~ . , data=data, pch=16, 
      col = ifelse(data$Bombed == TRUE, "tomato3", "black"), 
      main="Pairwise Relationships of Runner Features.")

# 3. THE HACK: Create a transparent, invisible plot over the entire canvas
par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(0, 0, type = 'n', bty = 'n', xaxt = 'n', yaxt = 'n')

# 4. Drop the legend onto this invisible layer
legend("bottom", legend = c("Safe (FALSE)", "Bombed (TRUE)"), 
       col = c("black", "tomato3"), pch = 16, 
       horiz = TRUE, bty = "n", inset = c(0, 0))

# 5. Reset your plot parameters to normal
par(oma = c(0, 0, 0, 0), fig = c(0, 1, 0, 1))
# Then we check for distribution of the data
par(mfrow=c(2,2))
hist(data$Carbs)
hist(data$Fluid)
hist(data$AveSpeed)
boxplot(data)
# From the histogram and the boxplot, we see that the features have ranges that vary greatly.
# This is bad for neural networks. We would have needed to rescale the data if we were using 
# the standard gradient descent. However, R's nlm uses a Newton like second order derivate optimisation
# algorithm, which implicitly takes care of the scaling

par(mfrow = c(1, 1))
softmax = function(Z_l){
  # This fucntion takes in a q x N matrix,
  # with z vector for each observation as columns
  # Outputs a q x N probability matrix
  
  # We first calculatethe max value for each column
  z_max = apply(Z_l, 2, max)
  # turn the max matrix into a diagonal matrix
  # Use Diagonal to prevent exp(0) = 1
  expZ_max = Diagonal(x = exp(-z_max))
  expZ_l = exp(Z_l)
  expDiff = expZ_l %*% expZ_max
  
  # Get the denominators 
  Denominator = Diagonal(x = 1/colSums(expDiff))
  
  S = expDiff %*% Denominator
  return(as.matrix(S))
}

Xu = cbind(data$AveSpeed, data$Sex)
Xv = cbind(data$Carbs, data$Fluid)

# One-hot encoding the response for two outputs as requested
Y = cbind(1 - data$Bombed, data$Bombed)

sig = function(A){
  return(tanh(A))
}

sig_out = function(A){
  return(softmax(A))
}

obj = function(Y, Y_hat){
  epsilon = 1e-8 # to prevent log(0)
  Y_hat = pmax(pmin(Y_hat, 1-epsilon), epsilon)
  return(sum(-(Y * log(Y_hat))))
}

neural_net = function(Xu, Xv, m, theta, nu, Y=NULL){
  # output: Y_hat: Raw probability output of the logistic function
  # output: pred: prediction value
  # output: E: evaluation error as required
  
  pu = dim(Xu)[2]
  pv = dim(Xv)[2]
  N = dim(Xu)[1]
  X = cbind(Xu, Xv)
  
  # 1st layer (m nodes per hemisphere)
  index = 1:(m*pu)
  W1u = matrix(theta[index], pu, m)
  index = max(index) + 1:(pv*m)
  W1v = matrix(theta[index], pv, m)
  W1 = as.matrix(bdiag(W1u, W1v))
  
  # 2nd layer (SPLIT BRAIN: 1 node per hemisphere)
  index = max(index) + 1:m
  W2u = matrix(theta[index], m, 1)
  index = max(index) + 1:m
  W2v = matrix(theta[index], m, 1)
  W2 = as.matrix(bdiag(W2u, W2v))
  
  # Output layer (Fully connected, 4 parameters)
  index = max(index) + 1:4
  W3 = matrix(theta[index], 2, 2)
  
  # Biases Layer 1
  index = max(index) + 1:m
  b1u = matrix(theta[index], m, 1)
  index = max(index) + 1:m
  b1v = matrix(theta[index], m, 1)
  b1 = rbind(b1u, b1v)
  
  # Biases Layer 2 (1 bias per hemisphere, 2 total)
  index = max(index) + 1:2
  b2 = matrix(theta[index], 2, 1)
  
  # Biases Output Layer (2 nodes, 2 biases)
  index = max(index) + 1:2
  b3 = matrix(theta[index], 2, 1)
  
  # Forward Pass
  ones = matrix(1, N, 1)
  A0 = as.matrix(t(X))
  A1 = sig(t(W1) %*% A0 + b1 %*% t(ones))
  A2 = sig(t(W2) %*% A1 + b2 %*% t(ones))
  Z_out = t(W3) %*% A2 + b3 %*% t(ones)
  
  # Output mapping
  Y_hat = t(sig_out(Z_out))
  pred = max.col(Y_hat) - 1
  
  # Evaluation
  E = NULL
  if (!is.null(Y)){
    base_error = obj(Y, Y_hat) / nrow(Y)
    penalty = (nu / nrow(Y)) * (sum(W1^2) + sum(W2^2) + sum(W3^2))
    E = base_error + penalty
  }
  
  return(list(Y_hat = Y_hat, pred = pred, E = E))
}

set.seed(2026)
m = 4
N = dim(Y)[1]
train_indices = sample(1:N, size=round(0.8*N))

Xtrain_u = Xu[train_indices,]
Xtrain_v = Xv[train_indices,]
Ytrain = as.matrix(Y[train_indices, ])

Xvalid_u = Xu[-train_indices,]
Xvalid_v = Xv[-train_indices,]
Yvalid = as.matrix(Y[-train_indices, ])

pu = dim(Xu)[2]
pv = dim(Xv)[2]
nparams = (pu * m) + (pv * m) + 2 * m  + 2 * m + 4 + 4
theta = runif(nparams, -1, 1)

nu_grid = exp(seq(-6,2,length
                  =40))
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

plot(errors ~ log(nu_grid), pch=16, col="firebrick4", type="b", 
     main="Validation Error vs. Regularization Parameter (Log nu).",
     xlab="Log(nu)", ylab="Validation Error", 
     las=1)
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

response_male = neural_net(XXumale, XXv, 4, theta_best_nu, 0)
response_female = neural_net(XXufemale, XXv, 4, theta_best_nu, 0)

plot(xxv2 ~ xxv1, col=color.gradient(response_female$Y_hat[,2]), pch=16, cex=0.8, 
     xlab="Carbohydrate intake (gram/hour)", ylab="Fluid intake (litre/hour)", 
     main="Female: probability of hitting the wall.")
legend("topright", title="Risk of Bombing",
       legend=c("High (Near 100%)", "Medium (~50%)", "Low (Near 0%)"), 
       col=c("skyblue1", "purple", "coral"), pch=16, bg="white", cex=0.7)

plot(xxv2 ~ xxv1, col=color.gradient(response_male$Y_hat[,2]), pch=16, cex=0.8, 
     xlab="Carbohydrate intake (gram/hour)", ylab="Fluid intake (litre/hour)", 
     main="Male: probability of hitting the wall.")
legend("topright", title="Risk of Bombing",
       legend=c("High (Near 100%)", "Medium (~50%)", "Low (Near 0%)"), 
       col=c("skyblue1", "purple", "coral"), pch=16, bg="white", cex=0.7)


# Empirical comparison 4-4 network

std_neural_net = function(X, m, theta, nu, Y=NULL){
  p = dim(X)[2]
  N = dim(X)[1]
  q = 2
  
  index = 1:(m*p)
  W1 = matrix(theta[index], p, m)
  index = max(index) + 1:(m*m)
  W2 = matrix(theta[index], m, m)
  index = max(index) + 1:(m*q)
  W3 = matrix(theta[index], m, q)
  
  index = max(index) + 1:m
  b1 = matrix(theta[index], m, 1)
  index = max(index) + 1:m
  b2 = matrix(theta[index], m, 1)
  index = max(index) + 1:q
  b3 = matrix(theta[index], q, 1)
  
  ones = matrix(1, N, 1)
  A0 = as.matrix(t(X))
  A1 = sig(t(W1) %*% A0 + b1 %*% t(ones))
  A2 = sig(t(W2) %*% A1 + b2 %*% t(ones))
  Z_out = t(W3) %*% A2 + b3 %*% t(ones)
  
  Y_hat = t(sig_out(Z_out))
  pred = max.col(Y_hat) - 1
  
  E = NULL
  if (!is.null(Y)){
    base_error = obj(Y, Y_hat) / nrow(Y)
    penalty = (nu / nrow(Y)) * (sum(W1^2) + sum(W2^2) + sum(W3^2))
    E = base_error + penalty
  }
  return(list(Y_hat = Y_hat, pred = pred, E = E))
}

# Setup the combined data
Xtrain_full = cbind(Xtrain_u, Xtrain_v)
Xvalid_full = cbind(Xvalid_u, Xvalid_v)

# Standard (4,4) has exactly 50 parameters
std_nparams = 50 
set.seed(2026)
std_theta = runif(std_nparams, -1, 1)
std_errors = rep(1, length(nu_grid))

cat("\n--- Training Standard Network ---\n")
for (i in 1:length(nu_grid)){
  nu = nu_grid[i]
  std_obj_train = function(theta_init) {
    result = std_neural_net(Xtrain_full, 4, theta_init, nu, Ytrain)
    return(result$E)
  }
  std_trained = nlm(std_obj_train, std_theta, iterlim=1000)
  std_valid_result = std_neural_net(Xvalid_full, 4, std_trained$estimate, nu)
  std_errors[i] = obj(Yvalid, std_valid_result$Y_hat) / nrow(Xvalid_full)
}

# Plot them together
plot(log(nu_grid), errors, type="b", col="firebrick4", pch=16, 
     main="Split-Brain vs Standard (4,4) Network",
     xlab="Log(nu)", ylab="Validation Error", ylim=range(c(errors, std_errors)))
lines(log(nu_grid), std_errors, type="b", col="skyblue", pch=16)
legend("topleft", legend=c("Split-Brain (40 params)", "Standard (50 params)"), 
       col=c("firebrick4", "skyblue"), lty=1, pch=16)