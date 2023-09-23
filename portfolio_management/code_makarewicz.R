###############################
# Exercise 1:Portfolio Analysis
###############################



# ****************
# Data Importation 
# ****************

# load the packages
library(quantmod)
library(xts)
library(PerformanceAnalytics)


# specify the ticker symbols
tickers <- c("IXC", "IDV", "SHY","^GSPC")

# ---IXC---: iShares Global Energy ETF
# ---IDV---: iShares International Select Dividend ETF
# ---SHY---: iShares 1-3 Year Treasury Bond ETF
# ---GSPC---: S&P 500 Index


# instantiate an empty matrix matrix to store all weekly prices
mat_cl_prices <- matrix()

# download each closing price and join it to the existing matrix
for(ticker in tickers) {
  # import OHLC data from yahoo finance for all available observations
  price <- getSymbols(Symbols = ticker, auto.assign = FALSE, src = "yahoo")
  # take the closing price
  cl_price <- Cl(price)
  # convert into weekly frequency
  week_cl_price <- to.period(cl_price, period = "weeks", k = 1)[, 4]
  # merge using all common observations (inner join) for each new downloaded price
  mat_cl_prices <- merge.xts(mat_cl_prices, week_cl_price, join = "inner") 
}



# ****************
# Data Exploration 
# ****************

# check the first and last rows
head(mat_cl_prices)
tail(mat_cl_prices)

# check dimensions: rows for obs and cols for ETFs
dim(mat_cl_prices)

# check column names
colnames(mat_cl_prices)

# check the frequency and the start/end dates
periodicity(mat_cl_prices)

# check the structure of the data
str(mat_cl_prices)



# *******************
# Data Pre-processing
# *******************

# drop first NA column 
mat_cl_prices$mat_cl_prices <- NULL
# replace not indicative column names
colnames(mat_cl_prices) <- c(tickers[1:3], "GSPC")
# now check the first rows
head(mat_cl_prices)


# check if any missing values to impute them, if no continue the analysis 
sum(is.na(mat_cl_prices))

# we can inspect the correlation matrix of ETFs to see if they are not too much correlated
cor(mat_cl_prices[, 1:3])


# log-return calculation
mat_returns <- Return.calculate(mat_cl_prices, method = "log")
# see teh first NA value
head(mat_returns)
# remove the first NA value as there is no prices before the first
mat_returns <- mat_returns[-1]
# display the first returns
head(mat_returns)


# ********************
# Statistical Analysis
# ********************

# package for time series analysis
library(tseries)

# summary statistics
summary(mat_returns)

# weekly mean and standard deviation
week_mean <- colMeans(mat_returns)
week_sd <- apply(mat_returns, 2, sd)
print(week_mean)
print(week_sd)

# inter quartile range
apply(mat_returns, 2, IQR)

# standard normal 95th quantile
qnorm_95 <- qnorm(p = 0.95, mean = 0, sd = 1)

bounds <- c(week_mean - qnorm_95 * week_sd,
            week_mean + qnorm_95 * week_sd) 
bounds_lab <- c("lower_bound", "upper_bound")

conf_interv_95 <- matrix(bounds, nrow = 2, ncol = 4,
                         dimnames = list(bounds_lab, tickers), byrow = TRUE)
print(conf_interv_95)


# central moments:
# annualized simple/arithmetic mean
ann_mean <- week_mean * 52
print(ann_mean)
# weekly and annualized standard deviation
ann_sd <- week_sd * sqrt(52)
print(ann_sd)
# skewness
skewness(mat_returns)
# excess kurtosis
kurtosis(mat_returns)

# check if skewness and kurtosis are statistically different from 0
jb_tests <- apply(mat_returns, 2, jarque.bera.test)

jb_tests$IXC
jb_tests$IDV
jb_tests$SHY
jb_tests$GSPC



# ******************
# Data Visualization
# ******************

# package for data visualization
library(ggplot2)
library(ggridges)
library(tidyr)


# Normalized time-series:
# take the first row for normalization
first_row <- coredata(mat_cl_prices[1, ])
# normalize at $100
mat_cl_prices_norm <- as.xts(t(apply(mat_cl_prices, 1, function(x) (x / first_row) * 100)))
# choose appropriate column names
colnames(mat_cl_prices_norm) <- tickers
colnames(mat_cl_prices_norm)[4] <- "GSPC"


# set a color scheme
color_pal <- c("#e2d810", "#d9138a", "#12a4d9", "#322e2f")
# plot the overlayed normalized closing prices
plot(x = as.zoo(mat_cl_prices_norm), xlab = "Date", ylab = "Price", main = "Closing Prices",
     col = color_pal, screens = 1)

# set a legend in the upper left hand corner to match color to return series
legend(x = "topleft", legend = colnames(mat_cl_prices_norm), 
       lty = c(1,1,1,1), col = color_pal)


# Histograms:
# set the common theme
theme <-theme_bw() + theme(plot.title = element_text(size = 15, hjust = 0.5, face = "bold")) 
theme_set(theme)

# IXC
p <- ggplot(mat_returns, aes(IXC)) + geom_histogram(aes(fill = ..count..),
                                                    bins = 30) + scale_fill_gradient(low = "blue", high = "cyan")
p + xlab("Return") + ggtitle("Histogram IXC") + labs(color = "Count") 

# IDV
p <- ggplot(mat_returns, aes(IDV)) + geom_histogram(aes(fill = ..count..),
                                                    bins = 30) + scale_fill_viridis_c(option = "cividis")
p +  xlab("Return") + ggtitle("Histogram IDV") + labs(color = "Count") 

# SHY
p <- ggplot(mat_returns, aes(SHY)) + geom_histogram(aes(fill = ..count..),
                                                    bins = 30) + scale_fill_viridis_c(option = "viridis")
p + xlab("Return") + ggtitle("Histogram SHY") + labs(color = "Count") 

# GSPC
p <- ggplot(mat_returns, aes(GSPC)) + geom_histogram(aes(fill = ..count..),
                                                     bins = 30) + scale_fill_viridis_c(option = "magma")
p + xlab("Return") + ggtitle("Histogram GSPC") + labs(color = "Count") 


# re-arange the dataframe as a long format instead of tidy format
mat_long <- gather(as.data.frame(mat_returns), key = ticker, value = ret, IXC, IDV, SHY, GSPC)
# plot kernel density estimates on the same graph
ggplot(mat_long, aes(x = ret, y = ticker, fill = ticker)) +
  geom_density_ridges() + xlab("Return") + ylab('Ticker') + ggtitle("Kernel Density Estimation")


# Box-plots:
# define function to plot statistical properties on the boxplots
get_box_stats <- function(y, upper_limit = max(mat_long$ret) * 1.15) {
  return(data.frame(
    y = 0.95 * upper_limit,
    label = paste(
      "Skew =", round(skewness(y),1), ",", "Ex Kurt =", round(kurtosis(y),1)
    )
  ))
}
# plot the box-plots
p <- ggplot(mat_long, aes(x = ret, y = ticker, fill = ticker)) + geom_boxplot() + labs(fill='Ticker Symbol') 
p + ggtitle("Boxplot") + xlab("Return") + ylab("Ticker") + labs(color = "Ticker Symbol") + stat_summary(fun.data = get_box_stats, geom = "text", hjust = 4, vjust = -3.5, size = 3)


# Q-Q plots:
# specify a facet grid of 2 rows and 2 columns
par(mfrow = c(2,2))
tickers <- colnames(mat_returns)
# set the color for each ETF
colors <- c("IXC" = "steelblue","IDV" = "seagreen", "SHY" = "deeppink4", "GSPC" = "orange")
# plot the Q-Q plots for each ETF using a for loop
for (ticker in tickers) {
  qqnorm(mat_returns[, ticker], xlab = "Standard Normal Quantile", ylab = "Empirical Quantile",
         main = paste("Q-Q plot", ticker))
  qqline(mat_returns[, ticker], col = colors[ticker], lwd = 2)
}


# ACF and PACF plots 
par(mfcol = c(2,4))
for (ticker in tickers) {
  acf(mat_returns[, ticker], main = paste("ACF raw returns", ticker,"30 lags"))
  pacf(mat_returns[, ticker], main = paste("PACF raw returns", ticker, "30 lags"))
}

# ACF and PACF plots for squared returns
par(mfcol = c(2,4))
for (ticker in tickers) {
  acf(mat_returns[, ticker]^2, main = paste("ACF squared returns", ticker,"30 lags"))
  pacf(mat_returns[, ticker]^2, main = paste("PACF squared returns", ticker, "30 lags"))
}



# Ljung-Box test with 30 lags
lj_box <- apply(mat_returns, 2, Box.test, lag = 30, type = "Ljung-Box")
lj_box$IXC
lj_box$IDV
lj_box$SHY
lj_box$GSPC





############################
# Exercise 2: Data modelling
############################

# packages for GARCH model and visualization
library(rugarch)
library(ggfortify)


# *********************************
# Jensen equation with GARCH errors
# *********************************

# see the volatility clusters
chart.RollingPerformance(mat_returns, FUN = "sd.annualized", width = 4 * 3,
                         scale = 52,
                         main = "Rolling 3-Month Annualized Standard Deviation",
                         legend.loc = "topright", 
                         colorset = rich6equal)


# -------------
# model fitting
# -------------

# specify the model (constant mean model with S&P 500 as external regressor)
general_garchspec <- ugarchspec(mean.model = list(armaOrder = c(0,0),
                                                  external.regressors = mat_returns$GSPC),
                                # standard GARCH(1,1) model for the variance model      
                                variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                                # normal distribution for the GARCH shocks
                                distribution.model = "norm")

# fit the model to the data for each ETF
garchfit_ixc <- ugarchfit(spec = general_garchspec, data = mat_returns$IXC)
garchfit_idv <- ugarchfit(spec = general_garchspec, data = mat_returns$IDV)
garchfit_shy <- ugarchfit(spec = general_garchspec, data = mat_returns$SHY)



# -------------------
# extracting features
# -------------------

# model summaries 
show(garchfit_ixc)
show(garchfit_idv)
show(garchfit_shy)

# optimal coefficients
coef(garchfit_ixc)
coef(garchfit_idv)
coef(garchfit_shy)

# persistences (alpha + beta GARCH parameters)
persistence(garchfit_ixc)
persistence(garchfit_idv)
persistence(garchfit_shy)

# unconditional variances
uncvariance(garchfit_ixc)
uncvariance(garchfit_idv)
uncvariance(garchfit_shy)


# -----------------------
# assessing model fitting
# -----------------------

# likelihoods
likelihood(garchfit_ixc)
likelihood(garchfit_idv)
likelihood(garchfit_shy)

# residuals 
resid_ixc <- residuals(garchfit_ixc)
resid_idv <- residuals(garchfit_idv)
resid_shy <- residuals(garchfit_shy)

# R2 intermediate calculations:
# sum of residuals squared
sum_squared_resid_ixc <- sum(resid_ixc^2)
sum_squared_resid_idv <- sum(resid_idv^2)
sum_squared_resid_shy <- sum(resid_shy^2)
# total sum of squares
total_sum_squares_ixc <- sum((mat_returns$IXC - mean(mat_returns$IXC))^2)
total_sum_squares_idv <- sum((mat_returns$IDV - mean(mat_returns$IDV))^2)
total_sum_squares_shy <- sum((mat_returns$SHY - mean(mat_returns$SHY))^2)

# R2 final calculation
R_squared_ixc <- 1 - (sum_squared_resid_ixc / total_sum_squares_ixc)
R_squared_idv <- 1 - (sum_squared_resid_idv / total_sum_squares_idv)
R_squared_shy <- 1 - (sum_squared_resid_shy / total_sum_squares_shy)

print(R_squared_ixc)
print(R_squared_idv)
print(R_squared_shy)

# verify with standard linear model
model_IXC <- lm(mat_returns$IXC ~ mat_returns$GSPC)
model_IDV <- lm(mat_returns$IDV ~ mat_returns$GSPC)
model_SHY <- lm(mat_returns$SHY ~ mat_returns$GSPC)

library(broom)
library(dplyr)
# similar to the previous one as GARCH model 
model_IXC %>% glance() %>% pull(r.squared)
model_IDV %>% glance() %>% pull(r.squared)
model_SHY %>% glance() %>% pull(r.squared)


# -----------------
# residual analysis
# -----------------

# standardized residuals (deameaned and divided by their standard deviation)
sd_resid_ixc <- residuals(garchfit_ixc, standardize = TRUE)
sd_resid_idv <- residuals(garchfit_idv, standardize = TRUE)
sd_resid_shy <- residuals(garchfit_shy, standardize = TRUE)

# time series of residuals 
mat_resid_merged <- cbind(sd_resid_ixc,sd_resid_idv, sd_resid_shy)
colnames(mat_resid_merged) <- c("IXC", "IDV", "SHY")
p <- autoplot(mat_resid_merged) + ggtitle('Time series of standardized residuals')
p + theme(plot.title = element_text(hjust = 0.5)) + xlab("Date") + ylab('Residual')

# time series of squared residuals (to highlight volatility clusters)
mat_sqrd_resid_merged <- cbind(sd_resid_ixc^2,sd_resid_idv^2, sd_resid_shy^2)
colnames(mat_sqrd_resid_merged) <- c("IXC", "IDV", "SHY")
p <- autoplot(mat_sqrd_resid_merged) + ggtitle('Time series of squared standardized residuals')
p + theme(plot.title = element_text(hjust = 0.5)) + xlab("Date") + ylab('Residual')


# autocorrelation diagnostic: (standard residuals)
# visual diagnostic
par(mfcol = c(2, 3))
for(ticker in tickers[1:3]) {
  acf(mat_resid_merged[, ticker], main = paste("ACF of residuals", ticker, "with 30 lags"))
  pacf(mat_resid_merged[, ticker], main = paste("PACF of residuals", ticker, "with 30 lags"))
}

# statistical diagnostic
Box.test(sd_resid_ixc, lag = 30, type = "Ljung-Box")
Box.test(sd_resid_idv, lag = 30, type = "Ljung-Box")
Box.test(sd_resid_shy, lag = 30, type = "Ljung-Box")


# autocorrelation diagnostic: (squared residuals)
# visual diagnostic
par(mfcol = c(2, 3))
for(ticker in tickers[1:3]) {
  acf(mat_sqrd_resid_merged[, ticker], main = paste("ACF of squared residuals", ticker, "with 30 lags"))
  pacf(mat_sqrd_resid_merged[, ticker], main = paste("PACF of squared residuals", ticker, "with 30 lags"))
}

# statistical diagnostic
Box.test(mat_sqrd_resid_merged[, "IXC"], lag = 30, type = "Ljung-Box")
Box.test(mat_sqrd_resid_merged[, "IDV"], lag = 30, type = "Ljung-Box")
Box.test(mat_sqrd_resid_merged[, "SHY"], lag = 30, type = "Ljung-Box")




# normality diagnostic:
# visual diagnostic
par(mfrow = c(1,3))
qqnorm(sd_resid_ixc, xlab = "Standard Normal Quantile",
       ylab = "Empirical Quantile", main = "Q-Q plot of residuals IXC")
qqline(sd_resid_ixc, col = "seagreen", lwd = 2)
# follow more closely the normal distribution
qqnorm(sd_resid_idv, xlab = "Standard Normal Quantile",
       ylab = "Empirical Quantile", main = "Q-Q plot of residuals IDV")
qqline(sd_resid_idv, col = "deeppink4", lwd = 2)
# large deviations from normal distribution in the tails
qqnorm(sd_resid_shy, xlab = "Standard Normal Quantile",
       ylab = "Empirical Quantile", main = "Q-Q plot of residuals SHY")
qqline(sd_resid_shy, col = "orange", lwd = 2)

# statistical diagnostic
jarque.bera.test(sd_resid_ixc)
jarque.bera.test(sd_resid_idv)
jarque.bera.test(sd_resid_shy)




# ************************
# Multivariate GARCH Model
# ************************

# multivariate garch package
library(rmgarch)

# rolling 6-month correlation
chart.RollingCorrelation(mat_returns[, 1:3], mat_returns$GSPC, width = 4 * 6,
                         scale = 52, main = "Rolling 6-Month Correlation",
                         legend.loc = "bottomright", colorset = rich8equal)

# -------------
# model fitting
# -------------

# same individual specification 
general_spec <- ugarchspec(mean.model = list(armaOrder = c(0,0)),
                           variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                           distribution.model = "norm")

# coerce the same individual specifications as a multivariate specification
multiv_spec <- multispec( replicate(4, general_spec) )

# dynamic conditional covariance specification class but without a VaR model for the mean
dcc_spec <- dccspec(multiv_spec, VAR = FALSE)

# fitting the model to the data
multiv_fit <- dccfit(dcc_spec, data = mat_returns)



# -------------------
# extracting features
# -------------------

# model summary
show(multiv_fit)

# information criterias
infocriteria(multiv_fit)

# joint likelihood
likelihood(multiv_fit)




# -----------------
# residual analysis 
# -----------------

# extract the residuals
resid <-residuals(multiv_fit)
resid_squared <- resid^2

# time series of residuals
p <- autoplot(resid) + ggtitle("Time Series of Residuals") + theme(plot.title = element_text(hjust = 0.5))
p + xlab("Date") + ylab('Residual')


# time series of squared residuals (to highlight volatility clusters)
p <- autoplot(resid_squared) + ggtitle('Time series of squared residuals')
p + theme(plot.title = element_text(hjust = 0.5)) + xlab("Date") + ylab('Residual')


# autocorrelation diagnostic:
# visual diagnostic
par(mfcol = c(2,4))
for (ticker in tickers) {
  acf(resid[, ticker], main = paste("ACF of residuals", ticker))
  pacf(resid[, ticker], main = paste("PACF of residuals", ticker))
}


# statistical diagnostic
lj_box <- apply(resid, 2, Box.test, lag = 30, type = "Ljung-Box")
lj_box$IXC
lj_box$IDV
lj_box$SHY
lj_box$GSPC



# normality diagnostic:
# visual diagnostic
par(mfrow = c(2,2))
colors <- c("seagreen", "deeppink4", "orange", "cyan")
names(colors) <- tickers
for (ticker in tickers) {
  qqnorm(resid[, ticker], xlab = "Standard Normal Quantile",
         ylab = "Empirical Quantile", main = paste("Q-Q plot of residuals", ticker))
  qqline(resid[, ticker], col = colors[ticker], lwd = 2)
}



# statistical diagnostic
for(ticker in tickers) {
  print(ticker)
  print(jarque.bera.test(resid[, ticker]))
}

# statistical properties of residuals
skewness(resid)
kurtosis(resid)




# ----------------
# conditional beta
# ----------------

# filtered dynamic conditional covariance array (the third dimensions gives the time index)
cov_matrix <- rcov(multiv_fit)

# time-varying betas (extract a time series of beta for each time index)
cond_var_gspc <- cov_matrix["GSPC", "GSPC", ]
tv_beta_ixc <- cov_matrix["IXC", "GSPC", ] / cond_var_gspc
tv_beta_idv <- cov_matrix["IDV", "GSPC", ] / cond_var_gspc
tv_beta_shy <- cov_matrix["SHY", "GSPC", ] / cond_var_gspc

# go back to xts to have dates in the x-axis
tv_beta_ixc <- as.xts(tv_beta_ixc, index(mat_returns))
tv_beta_idv <- as.xts(tv_beta_idv, index(mat_returns))
tv_beta_shy <- as.xts(tv_beta_shy, index(mat_returns))




# plot the time-varying betas
plot.zoo(tv_beta_ixc, main = "Conditional Beta IXC",
         xlab = "Date", ylab = "Beta", col = "aquamarine3")
plot.zoo(tv_beta_idv, main = "Conditional Beta IDV",
         xlab = "Date", ylab = "Beta", col = "tan1")
plot.zoo(tv_beta_shy, main = "Conditional Beta SHY",
         xlab = "Date", ylab = "Beta", col = "mediumvioletred")





###########################################
# Exercise 3: Standard Performance Measures
###########################################

# we assume a 0% risk-free rate and a 5% confidence level
risk_free_rate <- 0
conf_level <- 0.05


# *********************
# Standard Sharpe ratio
# *********************

# weekly Sharpe ratio
expected_returns <- colMeans(mat_returns)
excess_returns <- expected_returns - mean(risk_free_rate)
st_dev <- apply(mat_returns, 2, sd)
sharpe_ratio <- excess_returns / st_dev
print(sharpe_ratio)

# annualized Sharpe ratio assuming 52 trading weeks in a year
ann_expected_returns <- expected_returns * 52
ann_risk_free_rate <- risk_free_rate * 52
ann_st_dev <- sqrt(52) * st_dev
ann_sharpe_ratio <- (ann_expected_returns - mean(ann_risk_free_rate)) / ann_st_dev
print(ann_sharpe_ratio)




# ******************
# Smart Sharpe ratio
# ******************

# autocorrelation array
array_acf <- acf(mat_returns, lag.max = 1, plot = FALSE)$acf

# understand the structure to extract the first order autocorrelation 
print(array_acf)
# 3-dimensional array 
dim(array_acf)

# instantiate an empty matrix to store the results
rho <- matrix()
# extract the first order autocorrelation for each ETF
for(ticker in 1:4) {
  # extract the second row (the first row is order O) for each ETF
  rho[ticker] <- array_acf[2, ticker, ticker]
}

# compute the smart Sharpe ratio aka the long-run variance sharpe ratio
smart_sharpe_ratio <- sharpe_ratio * 2 / (2 + rho)
print(smart_sharpe_ratio)
# is the smart Sharpe ratio larger than the standard one ?
print(smart_sharpe_ratio > sharpe_ratio)
# show the relative difference 
print((smart_sharpe_ratio / sharpe_ratio) - 1)



# ****************
# Sharpe-VaR Ratio
# ****************

# ---------------
# gaussian method
# ---------------
# compute the standard normal 5th percentile
std_norm_quantile <- qnorm(p = conf_level, mean = 0, sd = 1)
# - sign in front of the VaR as it represents a loss
VaR_gauss_95 <- - (expected_returns + st_dev * std_norm_quantile)
sharpe_VaR_gauss_95 <- excess_returns / VaR_gauss_95

print(VaR_gauss_95)
print(sharpe_VaR_gauss_95)
# is the Sharpe-Gaussian VaR larger than the standard one ?
print(sharpe_VaR_gauss_95 > sharpe_ratio)
# show the relative difference
print((sharpe_VaR_gauss_95 / sharpe_ratio) - 1)


# ------------------
# historical  method
# ------------------
# number of observations
nb_obs <- nrow(mat_returns)
# sort the returns in ascending order
sorted_returns <- apply(mat_returns, 2, sort)
# take the appropriate index
idx_largest_losses <- nb_obs * conf_level
# extract the corresponding VaR
VaR_hist_95 <-  - sorted_returns[idx_largest_losses, ]
sharpe_VaR_hist_95 <- excess_returns / VaR_hist_95

print(VaR_hist_95)
print(sharpe_VaR_hist_95)
# is the Sharpe-hist VaR higher than the gaussian one ?
print((sharpe_VaR_hist_95 / sharpe_VaR_gauss_95) - 1)


# ---------------------
# cornish-fisher method
# ---------------------

# compute the 5% normal quantile
z_std_norm_5 <- std_norm_quantile
# define the coefficients for a confidence level of 5%
a <- 0.284
b <- 0.020
c <- 0.019
skew <- skewness(mat_returns)
kurt <- kurtosis(mat_returns)

# compute the cornish-fisher quantile
z_cf_5 <- z_std_norm_5 + a * skew + b * as.matrix(kurt) + c * as.matrix(skew)^2
rownames(z_cf_5) <- NULL
print(z_std_norm_5)
print(z_cf_5)
print(skew)
print(kurt)

VaR_cf_95 <- - (expected_returns + st_dev * z_cf_5)
sharpe_VaR_cf_95 <- excess_returns / VaR_cf_95
print(VaR_cf_95)
print(sharpe_VaR_cf_95)
# is the CF VaR higher than the gaussian VaR ?
print(VaR_cf_95 > VaR_gauss_95)


# compute the 1% normal quantile
z_std_norm_1 <- qnorm(p = 0.01, mean = 0, sd = 1)
# define the coefficients for a confidence level of 1%
a <- 1.425
b <- - 0.843
c <- 1.210
# compute the cornish-fisher quantile
z_cf_1 <- z_std_norm_1 + a * skew + b * as.matrix(kurt) + c * as.matrix(skew)^2
rownames(z_cf_1) <- NULL
print(z_std_norm_1)
print(z_cf_1)




# *****************
# Sharpe-CVaR Ratio
# *****************

# ---------------
# gaussian method
# ---------------
# compute the pdf
phi <- dnorm(z_std_norm_5, mean = 0, sd = 1)
CVaR_gauss_95 <- expected_returns + st_dev * (phi / conf_level)
sharpe_CVaR_gauss_95 <- excess_returns / CVaR_gauss_95
print(CVaR_gauss_95)
print(sharpe_CVaR_gauss_95)
# is the CVaR always larger than the VaR ? 
print(CVaR_gauss_95 > VaR_gauss_95)
# relative difference between CvaR and VaR
print((CVaR_gauss_95 /VaR_gauss_95) - 1)

# -----------------
# historical method
# -----------------

# consider the returns worst than the historical VaR at 95%
k_largest_losses <- sorted_returns[1:idx_largest_losses,]
# CvaR is the mean of these returns
CVaR_hist_95 <- - colMeans(k_largest_losses)
sharpe_CVaR_hist_95 <- excess_returns / CVaR_hist_95
print(CVaR_hist_95)
print(sharpe_CVaR_hist_95)
# is the hist CvaR larger than the gaussian one ?
print(CVaR_hist_95 > CVaR_gauss_95)
# relative difference between the hist CvaR and the gaussian
print((CVaR_hist_95 / CVaR_gauss_95) - 1)


# ---------------------
# cornish-fisher method
# ---------------------

y_05 <- st_dev * (phi / conf_level)
# define the coefficients for a confidence interval of 95%
m <- -0.2741
p <- -0.1225
q <- 0.0711

Y_05 <- y_05 * (1 + m * skew + p * skew^2 + q * kurt)
rownames(Y_05) <- NULL
CVaR_cf_95 <- expected_returns + Y_05
sharpe_CVaR_cf_95 <- excess_returns / CVaR_cf_95
print(CVaR_cf_95)
print(sharpe_CVaR_cf_95)
# is the cf CVaR larger than the normal CVaR ?
print(CVaR_cf_95 > CVaR_gauss_95)
# relative difference between CF CvaR and gaussian CVaR 
print((CVaR_cf_95 / CVaR_gauss_95) - 1)




################################
# Exercise 4: Draw-down Analysis
################################

# *********************
# Draw-down Calculation
# *********************

# window size of 2 years
window_size <- 2 * 52 

# number of observations
nb_obs <- nrow(mat_cl_prices)

# instantiate an empty matrix to store results
moving_max <- xts(matrix(NA, nrow = nb_obs, ncol = 4,
                         dimnames = list(index(mat_cl_prices),
                                         colnames(mat_cl_prices))),
                  index(mat_cl_prices))

# we have to compute the first iteration apart
moving_max[window_size, ] <- apply(mat_cl_prices[1:window_size], 2, max)
# find the maximum for the moving window one week ahead

for (i in 1:(nb_obs - window_size)) {
  # take the maximum of the window discarding the first obs and adding the next one
  moving_max[window_size + i, ] <- apply(mat_cl_prices[(i+1):(window_size + i), ], 2, max)
}

# matrix of draw-downs for all ETFs
mat_dd<- (mat_cl_prices / moving_max) - 1
print(mat_dd[1:5, ])
print(mat_dd[window_size:(window_size+5)])


# plot the underwater curve 
tsRainbow <- rainbow(4)
plot(as.zoo(mat_dd), xlab = "Date", ylab = "Drawdown", main = "Underwater curve",
     col = tsRainbow, screens = 1)
# set a legend in the bottom left corner to match color to return series
legend(x = "bottomright", legend = colnames(mat_cl_prices), 
       lty = c(1,1,1,1), col = tsRainbow)


# specify a facet grid of 2 rows and 2 columns
par(mfrow = c(2,2), cex = 0.8)
# vectors of colors 
tsRainbow <- rainbow(8)
# specify 2 colors for each graph
cols <- list("IXC" = tsRainbow[1:2], "IDV" = tsRainbow[3:4],
             "SHY" = tsRainbow[5:6], "GSPC" = tsRainbow[7:8])
tickers <- colnames(mat_returns)
# specify the perfect location for each plot
locations <- c("IXC" = "topright", "IDV" = "topright",
               "SHY" = "topleft", "GSPC" = "topleft")

for (ticker in tickers) {
  mat_temp <- cbind(mat_cl_prices[, ticker], moving_max[, ticker])
  colnames(mat_temp) <- c("Closing Price", "Moving Max")
  plot.zoo(as.zoo(mat_temp), xlab = "Date", ylab = "Price",
           col = cols[[ticker]], main = ticker, plot.type = "single")
  legend(locations[ticker], legend = colnames(mat_temp), col = cols[[ticker]], lty = c(1,1))
}




# ********************
# Performance Measures
# ********************

# confidence level
conf_level <- 0.05

# remove the first 3 years of NAs for further calculations
mat_dd <- na.omit(mat_dd)

# -----------
# ulcer index
# -----------

# compute the ulcer index
squared_dd <- mat_dd * mat_dd
ulcer_idx <- sqrt(colMeans(squared_dd))
print(ulcer_idx)
# compute the ulcer performance index  
ulcer_perf_idx <- excess_returns / ulcer_idx
print(ulcer_perf_idx)


# -----------------
# draw-down at risk
# -----------------

# sort drawdowns in ascending order
sorted_dd <- apply(mat_dd, 2, sort)
# extract the index of the largest drawdowns
idx_largest_dd <- conf_level * nb_obs
# extract that particular drawdown, namely the DaR
DaR_95 <- sorted_dd[idx_largest_dd, ]
print(DaR_95)


# -----------------------------
# conditional draw-down at risk
# -----------------------------

# compute the mean of the largest drawdowns beyond the DaR
CDaR_95 <- colMeans(sorted_dd[1:idx_largest_dd, ])
print(CDaR_95)


# -----------------
# average draw-down
# -----------------

# average drawdown
avg_dd <- colMeans(mat_dd)
print(avg_dd)




# draw-down distributions:
# IXC
p <- ggplot(mat_dd, aes(IXC)) + geom_histogram(aes(fill = ..count..), bins = 30) + scale_fill_viridis_c(option = "mako")
p <- p + xlab("Drawdown") + ylab("Count") + ggtitle("Drawdown  Distribution IXC") 
p <- p + geom_vline(aes(xintercept = DaR_95["IXC"], color = "blue"))
p <- p + geom_vline(aes(xintercept = CDaR_95["IXC"], color = "red") )
p <- p + geom_vline(aes(xintercept = avg_dd["IXC"], color = "green") )
p + scale_color_identity(name = "Risk measure",
                         breaks = c("blue", "red", "green"),
                         labels = c(paste("DaR 95:", format(round(DaR_95["IXC"], 2), nsmall = 2)), paste("CDaR 95:", format(round(CDaR_95["IXC"], 2), nsmall = 2)), paste("Avg Dd:", format(round(avg_dd["IXC"], 2), nsmall = 2))),
                         guide = "legend" ) + scale_linetype_manual(values=c("twodash", "dotted", "twodash"))


# IDV
p <- ggplot(mat_dd, aes(IDV)) + geom_histogram(aes(fill = ..count..), bins = 30) + scale_fill_viridis_c(option = "inferno")
p <- p + xlab("Drawdown") + ylab("Count") + ggtitle("Drawdown  Distribution IDV") 
p <- p + geom_vline(aes(xintercept = DaR_95["IDV"], color = "blue"))
p <- p + geom_vline(aes(xintercept = CDaR_95["IDV"], color = "red") )
p <- p + geom_vline(aes(xintercept = avg_dd["IDV"], color = "green") )
p + scale_color_identity(name = "Risk measure",
                         breaks = c("blue", "red", "green"),
                         labels = c(paste("DaR 95:", format(round(DaR_95["IDV"], 2), nsmall = 2)), paste("CDaR 95:", format(round(CDaR_95["IDV"], 2), nsmall = 2)), paste("Avg Dd:", format(round(avg_dd["IDV"], 2), nsmall = 2))),
                         guide = "legend" ) + scale_linetype_manual(values=c("twodash", "dotted", "twodash"))


# SHY
p <- ggplot(mat_dd, aes(SHY)) + geom_histogram(aes(fill = ..count..), bins = 30) + scale_fill_viridis_c(option = "rocket")
p <- p + xlab("Drawdown") + ylab("Count") + ggtitle("Drawdown  Distribution SHY") 
p <- p + geom_vline(aes(xintercept = DaR_95["SHY"], color = "blue"))
p <- p + geom_vline(aes(xintercept = CDaR_95["SHY"], color = "red") )
p <- p + geom_vline(aes(xintercept = avg_dd["SHY"], color = "green") )
p + scale_color_identity(name = "Risk measure",
                         breaks = c("blue", "red", "green"),
                         labels = c(paste("DaR 95:", format(round(DaR_95["SHY"], 2), nsmall = 2)), paste("CDaR 95:", format(round(CDaR_95["SHY"], 2), nsmall = 2)), paste("Avg Dd:", format(round(avg_dd["SHY"], 2), nsmall = 2))),
                         guide = "legend" ) + scale_linetype_manual(values=c("twodash", "dotted", "twodash"))


# GSPC
p <- ggplot(mat_dd, aes(GSPC)) + geom_histogram(aes(fill = ..count..), bins = 30) + scale_fill_viridis_c(option = "plasma")
p <- p + xlab("Drawdown") + ylab("Count") + ggtitle("Drawdown  Distribution GSPC") 
p <- p + geom_vline(aes(xintercept = DaR_95["GSPC"], color = "blue"))
p <- p + geom_vline(aes(xintercept = CDaR_95["GSPC"], color = "red") )
p <- p + geom_vline(aes(xintercept = avg_dd["GSPC"], color = "green") )
p + scale_color_identity(name = "Risk measure",
                         breaks = c("blue", "red", "green"),
                         labels = c(paste("DaR 95:", format(round(DaR_95["GSPC"], 2), nsmall = 2)), paste("CDaR 95:", format(round(CDaR_95["GSPC"], 2), nsmall = 2)), paste("Avg Dd:", format(round(avg_dd["GSPC"], 2), nsmall = 2))),
                         guide = "legend" ) + scale_linetype_manual(values=c("twodash", "dotted", "twodash"))




# -------------
# pitfall index
# -------------

# annual standard deviation
ann_std <- apply(mat_returns, 2, sd) * sqrt(52)
# the pitfall index as an extreme risk measure
pitfall_95 <- (- CDaR_95) / ann_std
print(pitfall_95)


# --------------
# penalized risk
# --------------

# penalized risk measure
penalized_risk <- ulcer_idx * pitfall_95
print(penalized_risk)


# --------------
# serenity ratio
# --------------

# the serenity ratio as risk-adjusted performance measure
ann_returns <- colMeans(mat_returns) * 52
serenity_ratio <- ann_returns / penalized_risk
print(serenity_ratio)






