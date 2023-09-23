
######################
# PART 1: OPTIMIZATION
######################


fn <- function(x){
  return( -x^4 + 3*x^3 + 9*x^2 + 23*x + 12 )
}
vx = seq(0,6,0.01)
plot(vx,fn(vx),type="l", col = "red", main = "Objective function")


# -----------
# Grid Search 
# -----------

x_min_GS <- 0
x_max_GS <- 6
vx_GS <- seq(from = x_min_GS, to = x_max_GS, by = 0.01)
x_opt_GS_1 <- 0
f_opt_GS_1 <- 0

# for loop method
for(x in vx_GS) {
  
  if(fn(x) > f_opt_GS_1) {
    x_opt_GS_1 <- x
    f_opt_GS_1 <- fn(x)
  }
}


# fast method
x_opt_GS_2<- vx_GS[which.max(fn(vx_GS))]
f_opt_GS_2 <- max(fn(vx_GS))

# we obtain the same results
all.equal(x_opt_GS_1, x_opt_GS_2)
all.equal(f_opt_GS_1, f_opt_GS_2)


#-------------
# QUASI-NEWTON 
#-------------

# need the opposite function because optim performs a minimization
neg_f <- function(x) {
  return(-1 * fn(x))
}

# initial starting point
x_init <- 0

res <- optim(par = x_init, fn  =neg_f, method = "BFGS")

# optimal x value
x_opt_QN <- res$par

# The optimized objective function value
f_opt_QN <- -1 * res$value


# close at 10^-7 than Grid Search 
all.equal(x_opt_GS_1, x_opt_QN)
all.equal(f_opt_GS_1, f_opt_QN)



#----------------------
# DIFERENTIAL EVOLUTION
#----------------------

# load the package
library("DEoptim")

res <- DEoptim(f=neg_f, lower = -5, upper= 5)

x_opt_DE <- res$optim$bestmem
f_opt_DE <- -1 * res$optim$bestval


# similar results
all.equal(x_opt_DE, x_opt_GS_1)
all.equal(x_opt_DE, x_opt_QN)

all.equal(f_opt_DE, f_opt_GS_2)
all.equal(f_opt_DE, f_opt_QN)




##########################
# PART 2: Initial Analysis
##########################

# first load the packages
library("quantmod")
library("ggplot2")
library("PerformanceAnalytics")
library('xts')
library("tseries")

# download Chevron directly in the environment 
getSymbols("CVX", from = "2004-01-01", to = "2022-08-31")

# download future oil in the variable oil to avoid forbidden named variable (=) 
oil <- Cl(getSymbols("CL=F", from = "2004-01-01", to = "2022-08-31", auto.assign = FALSE))

# check if any missing values
print(sum(is.na(oil)))
print(sum(is.na(CVX)))
# yes for oil, so impute them using last observation carried forward
oil <- na.locf(oil)


# data exploration for CVX
head(CVX)
tail(CVX)
class(CVX)
periodicity(CVX)


# extract the adjusted closing price since it is the most important feature for stocks
ad_cvx <- Ad(CVX)
colnames(ad_cvx) <- "Ad"


# facet grid to compare the two time series to see if there are correlated
par(mfrow = c(2,1), mex = 0.9, cex = 0.6)
plot(ad_cvx, col = "blue", main = "CVX", ylab = "Price", xlab = "Date")
plot(oil, col = "red", main = "Oil", ylab = "Price", xlab = "Date")


# plot using zoo package since xts is a subclass of class, it inherits all zoo methods
plot.zoo(ad_cvx, main = "Chevron's Adjusted Closing Price", ylab = "Price",
         xlab = "Date", col = "mediumvioletred")


# extract the prices from the years 2008 and 2020 using basic indexing
ad_cvx_2008 <- ad_cvx["2008"]
ad_cvx_2020 <- ad_cvx["2020"]

# plot using ggplot2 for data visualization to extract the worst 6 months
p <- ggplot(ad_cvx_2008, aes(x = index(ad_cvx_2008), y = Ad)) + geom_line(col = "aquamarine3") 
p + ggtitle("Chevron's Adjusted Closing Price for 2008") + xlab("Date") + ylab("Price")
p <- ggplot(ad_cvx_2020, aes(x = index(ad_cvx_2020), y = Ad)) + geom_line(col = "tan1") 
p + ggtitle("Chevron's Adjusted Closing Price for 2020") + xlab("Date") + ylab("Price")


# need the scope resolution operator (::) to access the first function from the xts class
# we extract the first 6 months from the beginning of the crashes
ad_cvx_2008_sub <- xts::first(ad_cvx_2008["2008-07/"], "6 months")
ad_cvx_2020_sub <- xts::first(ad_cvx_2020, "6 months")

# visualize the busts
plot.zoo(ad_cvx_2008_sub, main = "Chevron's Adjusted Closing Price from Jul 2008",
         ylab = "Price", xlab = "Date", col = "navyblue")
plot.zoo(ad_cvx_2020_sub, main = "Chevron's Adjusted Closing Price from Jan 2020",
         ylab  ="Price", xlab = "Date", col = "turquoise4")


# calculate log returns (dropping first NA values)
ret_cvx <- Return.calculate(ad_cvx, method = "log")[(-1),]
colnames(ret_cvx) <- "CVX_log_ret_full"
ret_cvx_2008_sub <- Return.calculate(ad_cvx_2008_sub, method = "log")[(-1),]
colnames(ret_cvx_2008_sub) <- "CVX_log_ret_2008_sub"
ret_cvx_2020_sub <- Return.calculate(ad_cvx_2020_sub, method = "log")[(-1),]
colnames(ret_cvx_2020_sub) <- "CVX_log_ret_2020_sub"


# plot histograms (more bins for the full period because we have much more obs)
p <- ggplot(ret_cvx, aes(CVX_log_ret_full)) + geom_histogram(aes(fill = ..count..),
                                                             bins = 50) + scale_fill_gradient(low = "cyan", high = "blue")
p + xlab("return") + ggtitle("Return Distribution full period")

p <- ggplot(ret_cvx_2008_sub, aes(CVX_log_ret_2008_sub)) + geom_histogram(aes(fill = ..count..),
                                                                        bins = 15) + scale_fill_gradient(low = "yellow", high = "red")
p +  xlab("return") + ggtitle("Return Distribution 2008's worst period")

p <- ggplot(ret_cvx_2020_sub, aes(CVX_log_ret_2020_sub)) + geom_histogram(aes(fill = ..count..),
                                                                        bins = 15) + scale_fill_gradient(low = "red", high = "blue")
p + xlab("return") + ggtitle("Return Distribution 2020's worst period")



# boxplots to emphasize extremes returns and quantiles
boxplot(ret_cvx, main = "Boxplot for the full period",
        col = "seagreen1",border = "seagreen4", ylab = "return")

boxplot(ret_cvx_2008_sub, main = "Boxplot 2008's worst period",
        ylab = "return", col = "blueviolet", border = "blue")

boxplot(ret_cvx_2020_sub, main = "Boxplot 2020's worst period",
        ylab = "return", col = "red2", border = "red4")



# Statistical properties

# Annualized mean
ann_mean_ret_cvx_full <- mean(ret_cvx) * 252
ann_mean_ret_cvx_2008_sub <- mean(ret_cvx_2008_sub) * 252
ann_mean_ret_cvx_2020_sub <- mean(ret_cvx_2020_sub) * 252

print(paste("Mean of daily returns for the full period:", ann_mean_ret_cvx_full))
print(paste("Mean of daily returns for the 2008's sub-period:", ann_mean_ret_cvx_2008_sub))
print(paste("Mean of daily returns for the 2020's sub-period:", ann_mean_ret_cvx_2020_sub))

# Annualized standard deviation
ann_sd_ret_cvx_full <- sd(ret_cvx) * sqrt(252)
ann_sd_ret_cvx_2008_sub <- sd(ret_cvx_2008_sub) * sqrt(252)
ann_sd_ret_cvx_2020_sub <- sd(ret_cvx_2020_sub) * sqrt(252)

print(paste("Standard deviation of daily returns for the full period:", ann_sd_ret_cvx_full))
print(paste("Standard deviation of daily returns for the 2008 period:", ann_sd_ret_cvx_2008_sub))
print(paste("Standard deviation of daily returns for the 2020 period:", ann_sd_ret_cvx_2020_sub))

# Third central moment 
skw_ret_cvx_full <- skewness(ret_cvx)
skw_ret_cvx_2008_sub <- skewness(ret_cvx_2008_sub)
skw_ret_cvx_2020_sub <- skewness(ret_cvx_2020_sub)

print(paste("Skewnes of daily returns for the full period:", skw_ret_cvx_full))
print(paste("Skewnes of daily returns for the 2008 period:", skw_ret_cvx_2008_sub))
print(paste("Skewness of daily returns for the 2020 period:", skw_ret_cvx_2020_sub))

# Fourth central moment (excess kurtosis is reported so kurtosis - 3)
krt_ret_cvx_full <- kurtosis(ret_cvx)
krt_ret_cvx_2008_sub <- kurtosis(ret_cvx_2008_sub)
krt_ret_cvx_2020_sub <- kurtosis(ret_cvx_2020_sub)

print(paste("Kurtosis of daily returns for the full period:", krt_ret_cvx_full))
print(paste("Kurtosis of daily returns for the 2008 period:", krt_ret_cvx_2008_sub))
print(paste("Kurtosis of daily returns for the 2020 period:", krt_ret_cvx_2020_sub))

# As third and fourth central moments different from 0, check if it is statistically significant
jarque.bera.test(ret_cvx)
jarque.bera.test(ret_cvx_2008_sub)
jarque.bera.test(ret_cvx_2020_sub)

# Summary statistics
summary(ret_cvx)
summary(ret_cvx_2008_sub)
summary(ret_cvx_2020_sub)


# Normal Q-Q plots
qqnorm(ret_cvx, xlab = "Standard Normal Quantiles", ylab = "Empirical Quantiles", main = "Standard Normal Q-Q plot full period")
qqline(ret_cvx, col = "steelblue", lwd = 2)

qqnorm(ret_cvx, xlab = "Standard Normal Quantiles", ylab = "Empirical Quantiles", main = "Standard Normal Q-Q plot 2008's sub-period")
qqline(ret_cvx, col = "seagreen", lwd = 2)

qqnorm(ret_cvx, xlab = "Standard Normal Quantiles", ylab = "Empirical Quantiles", main = "Standard Normal Q-Q plot 2020's sub-period")
qqline(ret_cvx, col = "deeppink4", lwd = 2)



# return series evolution
chartSeries(ret_cvx, theme = "white", name ="Log returns dynamics")


# rolling annualized simple mean (geometric = FALSE) return performance
chart.RollingPerformance(ret_cvx,
                         width = 22 * 3, scale = 252,
                         main = "Rolling 3-Month Annualized Mean Return", geometric = FALSE)

# rolling annualized volatility 
chart.RollingPerformance(ret_cvx, FUN = "sd.annualized", width = 22 * 3,
                         scale = 252, main = "Rolling 3-Month Annualized Standard Deviation return",
                         colorset = rich8equal)




# load rugarch package for GARCH models
library("rugarch")

# specify a simple GARCH(1,1) model with constant mean (armaOrder) and normal innovation
garchspec <- ugarchspec(mean.model = list(armaOrder = c(0, 0))
                                     , variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                                     distribution.model = "norm")

# fit the model to the data using the Maximum Likelihood estimation
garchfit <- ugarchfit(garchspec, data = ret_cvx)

# extract the fitted coefficients to write down the GARCH model
coefs<- coef(garchfit)
print(coefs)

# extract the square root of the unconditional variance so the standard deviation 
garch_unc_sd <- sqrt(uncvariance(garchfit))

# in-sample predictions
garch_vol <- sigma(garchfit)

# GARCH is a mean reverting to its unconditional volatility (red doted line)
plot.zoo(garch_vol, col = "blue", main  = "Daily Volatility forecast using a GARCH(1,1) model",
         xlab = "Date", ylab = "Standard Deviation")
abline(h = garch_unc_sd, col = "red", lwd = 1.5, lty = 2.0)


# 5% Value at Risk using the last in-sample prediction
last_vol_forecast <- tail(garch_vol, 1)
VaR_0.05 <- qnorm(p =0.05, mean = coefs["mu"], sd = last_vol_forecast)
last_ret <- tail(ret_cvx,1)

# compare it with the realized return
print(paste("Predicted 5% VaR is:", format(round(VaR_0.05*100,4), nsmall = 4), "%"))
print(paste("Realized return is:", format(round(last_ret*100,4), nsmall = 4), "%"))
      

# VaR with 2 * vol
VaR_0.05_2x <- qnorm(p = 0.05, mean = coefs["mu"], sd = 2*last_vol_for)

print(paste("Predicted 2*VaR is:", format(round(VaR_0.05_2x*100, 4), nsmal = 4), "%"))

portf_value <- 1000000
VaR_dollar <- portf_value * VaR_0.05
print(paste("$VaR for a portfolio of", format(portf_value, scientific = FALSE), "$",
            "is", format(round(VaR_dollar, 2), nsmall = 2), "$"))




####################
# Portfolio Analysis
####################

# ---APA---: APA Corporation (oil sector) stock
# ---ARLP---: Alliance Ressource Partners (oil sector) stock
# ---BKR---: Baker Hughes Company (oil sector) stock
# ---DINO---: HF Sinclair Corporation (oil sector) stock
# ---IXC---: iShares Global Energy ETF
# ---EFA---: iShares MSCI EAFE ETF
# ---DVY---: iShares Select Dividend ETF
# ---IBB---: iShares Biotechnology ETF
# ---QQQ---: Invesco QQQ Trust


# specify the 4 new oil stocks and the 5 ETFs
tickers <- c("APA", "ARLP", "BKR", "DINO", "IXC", "EFA", "DVY", "IBB", "QQQ")
start_date <- as.Date("2004-01-01")
end_date <- as.Date("2022-08-31")


# select close column (4th) from to.monthly because it converts into OHLC data
monthly_ad_cvx <- to.period(ad_cvx, period = "months", k = 1)[, 4]


# instantiate matrix to store prices for all instruments along with CVX 
mat_ad_prices <- matrix(monthly_ad_cvx)

# download each adjusted price and join it to the existing matrix
for(ticker in tickers) {
  price <- getSymbols(Symbols = ticker, from = start_date, to = end_date, auto.assign = FALSE)
  ad_price <- Ad(price)
  mat_ad_prices <- cbind(mat_ad_prices, to.period(ad_price, period = "months", k = 1)[, 4])
}

# download SP500 apart from the portfolio to use as a benchmark
sp500 <- Cl(getSymbols(Symbols = "^GSPC", from = start_date, to = end_date, auto.assign = FALSE))
sp500 <- to.period(sp500, period = "months", k = 1)[, 4]


# check structure (xts object, source ect ...)
print(str(mat_ad_prices))

# check dimensions (rows for monthly obs and columns assets)
print(dim(mat_ad_prices))

# check first and last observations
print(head(mat_ad_prices))
print(tail(mat_ad_prices))


# add column names as ticker symbols
colnames(mat_ad_prices) <- c("CVX", tickers)
colnames(sp500) <- "Sp500"

# check if any missing values to potentially impute them
print(sum(is.na(mat_ad_prices)))

# calculate simple monthly returns and drop first NA value
mat_returns <- Return.calculate(mat_ad_prices, method = "simple")[-1,]

# check summary statistics 
summary(mat_returns)


# package for dataframe manipulation and data visualization with ggplot2
library(dplyr)

mean_std_matrix <- as.data.frame(mat_returns) %>% summarize(mean = colMeans(mat_returns),
                                                        sd = apply(mat_returns, 2, FUN = "sd"))

# scatter plot with a regression line to see the positive relationship (i.e. the risk-reward tradeoff )
p <- ggplot(mean_std_matrix,aes(mean, sd)) + geom_point(col = "dodger blue",alpha = 0.5) + geom_smooth(method = "lm",se = FALSE,col = "red") + ggtitle("Mean return vs Standard Deviation") + ylab("Mean return") + xlab("Standard Deviation") 
p + annotate(geom = "text", x = mean_std_matrix$mean, y = mean_std_matrix$sd,
            label = colnames(mat_ad_prices))


# compute mean and standard deviation vectors
vMu <- colMeans(mat_returns)
vSd <- sd(mat_returns)

# compute covariance matrix
mCov <- cov(mat_returns)


# load package for quadratic optimization
library("quadprog")

# Number of assets
N <- ncol(mat_returns)

# optimization specification using the Markowitz's recommendation
# individual weight bounds (0 <= w <= 1)
lower_bound <- rep(0, N)
upper_bound <- rep(1, N)

dvec <- rep(0, N) # minimizing variance
Dmat <- mCov # semi positive definite cov matrix

At <- rbind(rep(1, N), # full investment
            vMu, # target return = individual expected return
            diag(rep(1,N)), # each weight >= 0
            diag(rep(-1, N))) # each -weights >= -1 

Amat <- t(At)

# grid to test out all return combinations from lowest to largest
target.mu.grid <- seq(min(vMu), max(vMu), length.out = 100)  

# define the weight matrix to store all optimized weights
weight.matrix <- matrix(NA, nrow = length(target.mu.grid), ncol = N)

for(i in 1:length(target.mu.grid)) {
  # the values on the right side of equal/ ineq constraints
  bound <- c(1, target.mu.grid[i] , lower_bound, -upper_bound)
  
  opt_weight <- try(solve.QP(Dmat = mCov, dvec = dvec, 
                        Amat = Amat, bvec = bound, meq = 2)$solution, silent = TRUE)
  
  if(class(opt_weight) != "try-error") {
    weight.matrix[i,] <- opt_weight
  }
}


# remove any missing values
weight.matrix <- na.omit(weight.matrix)

# compute the expected mean and standard deviation values from optimized weights
# using vectorized version with apply by passing an anonymous function 
sd.optimal <- apply(weight.matrix, 1, FUN = function(weight) sqrt(t(weight) %*% mCov %*% weight))
mu.optimal <- apply(weight.matrix, 1, FUN = function(weight) t(weight) %*% vMu)

# compute the vector of standard deviation of the assets
vSd <- sqrt(diag(mCov))

# plot the assets
plot(vSd,vMu, col = "gray",
     xlab = "Standard deviation (monthly)",
     ylab = "Expected return (monthly)", 
     xlim = c(0, max(vSd) + 0.01), 
     ylim = c(0,max(vMu)+0.005),
     las = 1)
text(vSd, vMu, labels = colnames(mat_returns), cex = 0.7)


# plot the optimized portfolios
lines(sd.optimal, mu.optimal, col = "blue", lwd = 2)


# index of the min var portfolio
idx_min_var <- which.min(sd.optimal)

# apply a mask (matrix of TRUE/FALSE) to identify efficient portfolios
mask_eff_front <- (mu.optimal >= mu.optimal[idx_min_var])

# indicate in red the efficient portfolios
lines(sd.optimal[mask_eff_front],  mu.optimal[mask_eff_front], col = "red", lwd = 2)



# ------------------
# Max weights of 15%
# ------------------

# define the weight constrained matrix to store all optimized weights
constr_weight_matrix <- matrix(NA, nrow = length(target.mu.grid), ncol = N )

# define new upper bound
upper_bound.15 <- rep(0.15,N) 


for(i in 1:length(target.mu.grid)) {
  
  bound <- c(1, target.mu.grid[i] , lower_bound, -upper_bound.15)
  opt_weight <-  try(solve.QP( Dmat = mCov, dvec = dvec, 
                        Amat = Amat, bvec = bound, meq = 2)$solution , silent=TRUE)
  
  if(class(opt_weight) != "try-error"){
    constr_weight_matrix[i,] <- opt_weight
  }
}

# remove the NAs
constr_weight_matrix <- na.omit(constr_weight_matrix)

# compute the mean and sd values of optimized weights
constr_sd_optimal <- apply(constr_weight_matrix, 1,
                           FUN = function(w) sqrt( t(w) %*% mCov %*% w))
constr_mu_optimal <- apply(constr_weight_matrix ,1,
                           FUN = function(w) t(w) %*% vMu)

# index of the minimum variance (MV) portfolio
idx_MV <- which.min(constr_sd_optimal)

# apply a mask to filter efficient portfolios
mask_eff_front <- (constr_mu_optimal >= constr_mu_optimal[idx_MV])
lines(constr_sd_optimal[mask_eff_front],
      constr_mu_optimal[mask_eff_front], col = "purple", lwd = 2)


legend("topright", legend=c("Unconstrained Effecicent Frontier",
                            "Unconstrained Inefficicent Frontier",
                            "Constrained Efficient Frontier"),
       col = c("red","blue", "purple"), lty = c(1,1,1), cex=0.5)




# weights for the unconstrained eff frontier for the MV portfolio
idx_unc_sd <- which.min(sd.optimal)
barplot(weight.matrix[idx_unc_sd,], names = colnames(mat_returns), col = "turquoise3",
        ylab = "Weight", main = "Weights Minimum Variance portfolio Unconstrained",)

# weights for the constrained efficient frontier
weights_MV <- constr_weight_matrix[idx_MV,]

weights_Max_Sharpe <- constr_weight_matrix[which.max(constr_mu_optimal / constr_sd_optimal), ]


barplot(weights_MV, names = colnames(mat_returns), ylab = "Weight",
        main = "Weights Minimum Variance portfolio Constrained", col = "violetred3")
barplot(weights_Max_Sharpe, names = colnames(mat_returns), ylab = "Weight",
        main = "Weights Maximum Sharpe ratio portfolio Constrained", col = "seagreen3")



#------------
# Backtesting
#------------

start_date_estim <- as.Date("2004-01-01")
end_date_estim <- as.Date("2010-12-31")


# split the data into in-sample and out-of-sample to avoid look ahead bias
mat_returns_estim <- window(mat_returns, start = start_date_estim, end = end_date_estim)
vMu_estim <- colMeans(mat_returns_estim)
mCov_estim <- cov(mat_returns_estim)

# define new target grid with VMu_estim and mCov_estim
target_mu_grid_estim <- seq(min(vMu_estim), max(vMu_estim), length.out = 100)  

constr_weight_matrix_estim <- matrix(NA, nrow = length(target_mu_grid_estim), ncol = N )


# find the optimal weights from the estimation period
for(i in 1:length(target_mu_grid_estim)) {
  
  bound <- c(1, target_mu_grid_estim[i] , lower_bound, -upper_bound.15)
  opt_weight <- try(solve.QP(Dmat = mCov_estim, dvec = dvec, 
                               Amat = Amat, bvec = bound,meq = 2)$solution, silent=TRUE)
  
  if(class(opt_weight) != "try-error"){
    constr_weight_matrix_estim[i,] <- opt_weight
  }
}

# remove the NAs
constr_weight_matrix_estim <- na.omit(constr_weight_matrix_estim)

# compute the mean and sd values of optimized weights
constr_sd_optimal_estim <- apply(constr_weight_matrix_estim, 1,
                                 FUN = function(weight) sqrt(t(weight) %*% mCov_estim %*% weight))
constr_mu_optimal_estim <- apply(constr_weight_matrix_estim ,1,
                                 FUN = function(weight) t(weight) %*% vMu_estim)

# find optimal portfolios
idx_MV_estim <- which.min(constr_sd_optimal_estim)
weights_MV_estim <- constr_weight_matrix_estim[idx_MV_estim, ]

idx_SR_estim <- which.max(constr_mu_optimal_estim/constr_sd_optimal_estim)
weights_Max_SR_estim <- constr_weight_matrix_estim[idx_SR_estim, ]




# define the evaluation sample one day ahead the end date of estimation
# to avoid look ahead bias
mat_returns_oos <- mat_returns["2011-01-01/"]

# calculate return portfolio from weights optimized on the estimation sample
mat_returns_portf_min_var_oos <- Return.portfolio(mat_returns_oos, weights = weights_MV_estim,
                                             rebalance_on = "months")
mat_returns_portf_max_sharpe_oos <- Return.portfolio(mat_returns_oos, weights = weights_Max_SR_estim,
                                                rebalance_on = "months")


# need to compute portfolio return for the estimation period so as to compare with evaluation sample
mat_returns_portf_min_var_estim <- Return.portfolio(mat_returns_estim, weights = weights_MV_estim,
                                               rebalance_on = "months")
mat_returns_portf_max_sharpe_estim <- Return.portfolio(mat_returns_estim, weights = weights_Max_SR_estim,
                                                  rebalance_on = "months")


# merge rows from the estimation and evaluation sample
# to compare with EW portfolio and sp500 for the full period
return_min_var_full <- rbind(mat_returns_portf_min_var_estim, mat_returns_portf_min_var_oos)
return_max_sharpe_full <- rbind(mat_returns_portf_max_sharpe_estim, mat_returns_portf_max_sharpe_oos)



# compute sp500 simple return series to compare with optimized portfolios
return_sp500 <- Return.calculate(sp500, method = "simple")[-1,]


# construct equally weighted portfolio also with monthly rebalancing
return_portf_ew <- Return.portfolio(mat_returns, weights = rep(1/N, N), rebalance_on = "months")


# merge returns to analyze relative performance
mat_returns_portf_merged<- cbind(return_min_var_full, return_max_sharpe_full,
                          return_portf_ew, return_sp500)

colnames(mat_returns_portf_merged) <- c("Min_Var", "Max_Sharpe","Eq_Weighted", "Sp500")



# ------------------------------------------------------------------
# Relative performance comparing all strategies for the full period
# ------------------------------------------------------------------


# annualized mean, standard deviation and sharpe ratio
table.AnnualizedReturns(mat_returns_portf_merged)


# evolution of 1$ invested at the beginning of the period
chart.CumReturns(mat_returns_portf_merged, wealth.index = TRUE, legend.loc="topleft",
                 main = "Cumulative return")

# rolling window of 2 years for annualized mean, standard deviation and sharpe ratio
charts.RollingPerformance(mat_returns_portf_merged, width = 24, legend.loc="topleft", scale = 12)

# left part of the return distribution analysis 
SemiDeviation(mat_returns_portf_merged)
SortinoRatio(mat_returns_portf_merged)


# analyze third and fourth central moment to check deviation from normality 
skewness(mat_returns_portf_merged)
kurtosis(mat_returns_portf_merged)
# test if statistically significant to take these features to compute ES
apply(mat_returns_portf_merged, 2, FUN = "jarque.bera.test")


# tail risk measures
VaR(mat_returns_portf_merged, p = 0.975)
ES(mat_returns_portf_merged, p = 0.975, method = "gaussian")
ES(mat_returns_portf_merged, p = 0.975, method = "modified")


# drawdown analysis for extremes losses
chart.Drawdown(mat_returns_portf_merged, legend.loc="bottomleft", main = "Drawdown plot")
# for the worst drawdown
maxDrawdown(mat_returns_portf_merged)
CalmarRatio(mat_returns_portf_merged)

# for mean and extreme drawdowns
ulcer_index <- UlcerIndex(mat_returns_portf_merged)
cond_drawdown <- CDD(mat_returns_portf_merged)
ann_std <- StdDev.annualized(mat_returns_portf_merged)

# pitfall and penalized risk metric
pitfall_indicator <- cond_drawdown / ann_std
penalized_risk <- pitfall_indicator * ulcer_index



# implementing the Serenity ratio function
SerenityRatio <- function(R, p = 0.95, geometric = FALSE, ...) {
  
  #'@description Compute the Serenity ratio as the ratio of the 
  #'expected return divided by a penalized risk measure that allows
  #'to consider drawdown risk
  #'
  #'@param R an xts, vector, matrix, data frame, timeSeries or zoo object of asset returns
  #'@param p confidence level for calculation, defaults to 95%
  #'@param geometric compute geometric mean or the simple arithmetic mean, defaults FALSE
  #'@param ... any other arguments
  #'
  #'@references R.Bagneaulot de Beville, R.Gelrubin, E.Lindet and C.Chevalier. An Alternative Portfolio Theory
  #'https://www.keyquant.com/Download/GetFile?Filename=%5CPublications%5CKeyQuant_WhitePaper_APT_Part1.pdf
  
  
  ulcer_index <- UlcerIndex(R, ...)
  
  cond_drawdown <- CDD(R, p = p, ...)
  ann_std <- StdDev.annualized(R, ...)
  
  pitfall_indicator <- cond_drawdown / ann_std
  
  penalized_risk <- ulcer_index * pitfall_indicator
  mean_return <- Return.annualized(R, geometric = geometric, ...)
  
  return(mean_return / penalized_risk)
  
}


SerenityRatio(mat_returns_portf_merged)





# --------------------------------------------------
# Absolute performance (estim vs evaluation samples)
# --------------------------------------------------

# min var
table.AnnualizedReturns(mat_returns_portf_min_var_estim)
table.AnnualizedReturns(mat_returns_portf_min_var_oos)

SerenityRatio(mat_returns_portf_min_var_estim)
SerenityRatio(mat_returns_portf_min_var_oos)


# max Sharpe ratio
table.AnnualizedReturns(mat_returns_portf_max_sharpe_estim)
table.AnnualizedReturns(mat_returns_portf_max_sharpe_oos)

SerenityRatio(mat_returns_portf_max_sharpe_estim)
SerenityRatio(mat_returns_portf_max_sharpe_oos)













