# fixing seed of the RNG for reproducibility
set.seed(123)

# download package for exception handling
library(assertive)

v_data_return <- rnorm(12, 0.02, 0.03)




#######################################
# EXERCICE 1: FINANCIAL DATA MANAGEMENT
#######################################


# implement draw down function with one data argument
compute_draw_down <- function(vect_ret) {
  
  #'@description Compute peak-to-through decline from a previous top
  #'@param vect_ret vector of returns 
  #'@examples 
  #'draw_downs <- compute_draw_down(vector_returns)
  
  # type constraint
  assert_is_numeric(vect_ret)
  
  # replace any missing values by the last observation carried forward method
  if(any(is.na(vect_ret))) na.locf(vect_ret)
  
  # initialize an empty vector to fill in in the for loop
  vect_drawdown <- rep(NA, length = length(vect_ret))
  
  # we must manually compute the first iteration so as to avoid using the
  # if else in the for loop
  if(vect_ret[1] < 0.0) {
    # drawdown is reported as a negative percentage number
    vect_drawdown[1] <- vect_ret[1]
  } else {
    vect_drawdown[1] <- 0.0
  }
  # entering the for loop using the previous drawdown
  for(i in 2:length(vect_ret)) {
    # drawdown is a cumulative negative return
    vect_drawdown[i] <- vect_drawdown[i-1] + vect_ret[i]
    
    # if the current return is positive then report 0
    # as drawdown is a negative number
    if(vect_drawdown[i] > 0) {
      vect_drawdown[i] <- 0.0
    }
  }
  return(vect_drawdown)
}

# sanity check
print(v_data_return)
print(compute_draw_down(v_data_return))


# implement max drawdown function with one data argument
compute_max_draw_down <- function(vect_return) {
  
  #'@description Compute the maximum drawdown from the compute_drawdown function 
  #'@param vect_return vector of returns to compute the maximum drawdown
  
  vect_drawdowns <- compute_draw_down(vect_return)
  return(min(vect_drawdowns))
}


# shortcut using lambda function
compute_max_draw_down_lambda <- function(vect_return) min(compute_draw_down(vect_return))

# we obtain identical results
print(compute_max_draw_down(v_data_return))
print(compute_max_draw_down_lambda(v_data_return))





#####################################
# EXERCICE 2: PERFORMANCE MEASUREMENT
#####################################



# implement expected shortfall function with 1 data and 3 default detail arguments
compute_ES <- function(vect_returns, method = "historical", p  = 0.90, interp = FALSE) {
  
  #'@description 
  #' Compute the Expected Shortfall using either the historical 
  #' method as the mean of the k largest losses from p or the parametric approach
  #' using the gaussian distribution
  #' 
  #' For instance, for the historical method if we had 1000 return observations and p = 0.90
  #' then we would compute the mean of the 1000*(1-0.90) = 10 largest losses 
  #' 
  #' 
  #'@param vect_returns the vector of returns
  #'@param method either "historical" or "gaussian", defaults to "historical"
  #'@param p confidence level for calculation, defaults to 90%
  #'@param interp lienar interpolation for decimal indices, defaults to FALSE
  #' 
  #'@examples 
  #'es_hist <- compute_ES(vect_ret)
  #'es_gauss.95 <- compute_ES(vect_ret, "gaussian", p = 0.95, interp = TRUE)

  
  
  
  # the code will either return the correspond ES or print an error message
  # for case values different than gaussian and historical
   if(method == "gaussian") {
      mean_returns <- mean(vect_returns)
      quantile_std_norm <- qnorm(p, mean = 0, sd = 1)
      sd_returns <- sd(vect_returns)
      expected_shortfall_gauss <- mean_returns - sd_returns * (dnorm(quantile_std_norm)) / (1 - p)
  
      # the function call ends as soon as a return command is encountered
      return(expected_shortfall_gauss)
  
      } else if (method == "historical") {
        # sort returns from ascending order 
        vect_returns_sorted <- sort(vect_returns)
        size <- length(vect_returns)
        idx_k_largest_losses <- (1-p) * size
        
        if(interp == TRUE) {
          idx_floor <- floor(idx_k_largest_losses)
          idx_ceil <- ceiling(idx_k_largest_losses)
          es_floor <- mean(vect_returns_sorted[1:idx_floor])
          es_ceil <- mean(vect_returns_sorted[1:idx_ceil])
          
          expected_shortfall_hs_interp <- es_floor + (es_ceil - es_floor) / (idx_ceil - idx_floor) * (idx_k_largest_losses - idx_floor)
          
          return(expected_shortfall_hs_interp)
          
        } else {
          # round to the closest integer (even though indexing with decimals is possible in R)
          rounded_idx_k_largest_losses <- round(idx_k_largest_losses)
          k_largest_losses <- vect_returns_sorted[1:idx_k_largest_losses]
          expected_shortfall_hs <- mean(k_largest_losses)
          return(expected_shortfall_hs)
        }
        
      } else {
        # simple exception handling 
        print("Not valid case name, it should be either 'historical' or 'gaussian'" )
      }
}

# sanity check
print(v_data_return)
print(compute_ES(v_data_return, "gaussian"))
print(compute_ES(v_data_return, "historical"))
print(compute_ES(v_data_return, "historical", interp = TRUE))




####################################
# EXERCICE 3: PORTFOLIO OPTIMIZATION
####################################



# implement simulation weights function with 2 default arguments
compute_simul_weights <- function(n_assets = 3, nb_simulations = 10000){
  
  #'@description Compute long-only as well as full-investment simulated weights  
  #'
  #'@param n_assets number of assets to simulate, defaults to 3
  #'@param nb_simulations number of simulations to perform, defaults to 10000
  #'
  #'@examples
  #'using default arguments:
  #'sim_weights_1 <- compute_simul_weights()
  #'using other arguments:
  #'sim_weights_2 <- compute_simul_weights(n = 100, nb_simulations = 10000)

  
  # type constraint
  assert_is_numeric(n_assets)
  assert_is_numeric(nb_simulations)
  
  # range constraint
  if(n_assets < 0 | nb_simulations < 0) {
    # throw an error
    stop("n_assets or nb_simulations contain/s negative value/s!.")
    
  } else {
    
    # create an empty matrix to store results
    mat_pos_weights <- matrix(rep(NA, length = n_assets * nb_simulations),
                              nrow = n_assets, ncol = nb_simulations) 
    
    for(simul in 1:nb_simulations) {
      
      # ensures positive weights
      col_pos_random_weights <- abs(runif(n_assets))
      # ensures sum of weights = 1
      mat_pos_weights[, simul] <- col_pos_random_weights * (1 / sum(col_pos_random_weights))
      
    }
    # function call ends as soon as a return command is seen
    return(mat_pos_weights)
    
 }
  
}

# sanity check  
mat_weights <- compute_simul_weights()

# check correct matrix dimensions (3x10000)
print(dim(mat_weights))

# ensure positive weights 
print(any(mat_weights < 0))
# ensure sum of weights = 1
sum_weights <- colSums(mat_weights)
print(any(sum_weights != 1))

# check why some do not sum to one
# (might be rounded errors as we've summed float weights to obtain an integer of 1)
sum_weights[which(sum_weights != 1)][1:10]

# indeed they all sum to 1 (as all.equal uses a tolerance error)
all.equal(sum_weights, rep(1, length(sum_weights)))




# for testing if we have to replace the simulated weights by restricted ones
test_weights <- function(vect_weights) {
  
  if(all(vect_weights > 0.05) & all(vect_weights < 0.95)) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}



compute_simul_weights_restricted <- function(restricted = TRUE) {
  
  #'@description Compute simulated weights from the compute_simul_weights function 
  #'but allow to restrict each individual weight between 5% and 95% for more
  #'diversified portfolios
  #'
  #'@param restricted compute restricted weights, defaults to TRUE
  #'
  #'@examples
  #'weights_restr<- compute_simul_weights_restricted()
  #'weights_unrestr <- compute_simul_weights_restricted(restricted = FALSE)
  
  
  
  # instantiate unrestricted portfolio weights
  mat_sim_weights <- compute_simul_weights()
  
  # test if restricted, if yes then iterate all columns , else return the unrestricted weights
  if(restricted == TRUE) {
    # iterate on all unconstrained simulated weights
    for(col in 1:ncol(mat_sim_weights)) {
      
      # if any asset weight < 0.05 or > 0.95, then recreate for this column random samples
      # from the restricted unif dist and ensure sum of weights = 1
      if(test_weights(mat_sim_weights[, col]) == FALSE) {
        
        # use brute force approach (do-while) until portfolio
        # criterias are fullfiled
        repeat {
        vect_restr_weights <- runif(3, min = 0.05, max = 0.95) 
        
        mat_sim_weights[, col] <- vect_restr_weights/ sum(vect_restr_weights)
        
        if(test_weights(mat_sim_weights[, col]) == TRUE) break
        }
       }
     }
  }
  return(mat_sim_weights)
}

# sanity check

# inspect execution time
start_time <- Sys.time()
mat_weights_restricted <- compute_simul_weights_restricted()
end_time <- Sys.time()

print(paste("Elapsed time:", end_time - start_time))

# compare it with the original version (approx 3 times longer)
start_time <- Sys.time()
mat_weights<- compute_simul_weights()
end_time <- Sys.time()

print(paste("Elapsed time:", end_time - start_time))


# check some weights distribution
print(mat_weights_restricted[,1:10])

# ensure sum of weights = 1
sum_weights_restricted <- colSums(mat_weights_restricted)
print(any(sum_weights_restricted != 1))

sum_weights_restricted[which(sum_weights_restricted != 1)][1:10]

all.equal(sum_weights_restricted, rep(1, length(sum_weights_restricted)))


# ensure lower and upper bounds are respected
print(any(mat_weights_restricted > 0.95))
print(any(mat_weights_restricted < 0.05))










