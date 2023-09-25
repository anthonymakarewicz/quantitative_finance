#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>
using namespace std;


/* A simple implementation of the Box-Muller algo used to generate gaussian random numbers
 * necessary for the Monte Carlo method
 */
 double gaussian_box_muller() {
	 
	 double x = 0.0;
	 double y = 0.0;
	 double euclid_sq = 0.0;
	 
	 // Continue generating 2 uniform r.v until the square of their euclidean distance is less than 1
	 do {
		 
		 x = 2.0 * rand() / static_cast<double>(RAND_MAX) -1;
		 y = 2.0 * rand() / static_cast<double>(RAND_MAX) -1;
		 
		 euclid_sq = x * x + y * y;
		 
	 } while(euclid_sq >= 1.0);
	 
	 return x * sqrt(-2 * log(euclid_sq) / euclid_sq);
	 
 }
 
 // Pricing a european vanilla call option with the Monte Carlo method
 double compute_call_price_mc(const int& N, const double& S,
                          const double& K, const double& r, const double& sig,
                          const double& T) {
							  
		double S_adjust = S * exp(T * (r - 0.5 * sig * sig));
		double S_cur = 0.0;
		double payoff_sum = 0.0;
		
		for(int i = 0; i < N; ++i) {
			
			double gauss_bm = gaussian_box_muller();
			
			S_cur = S_adjust * exp(sqrt(T * sig * sig)* gauss_bm);
			payoff_sum += std::max(S_cur - K, 0.0);
			
		}				
		
		return (payoff_sum/static_cast<double>(N)) * exp(-r*T);
		
	}
	
	
 // Pricing a european vanilla put with the Monte Carlo method
 double compute_put_price_mc(const int& N, const double& S, const double& K,
                              const double& r, const double& sig, const double& T) {
								  	  
		double S_adjust = S * exp(T * (r - 0.5 * sig * sig));
		double S_cur = 0.0;
		double payoff_sum = 0.0;
		
		for(int i = 0; i < N; ++i) {
			
			double gauss_bm = gaussian_box_muller();
			S_cur = S_adjust * exp(sqrt(sig * sig * T) * gauss_bm);
			payoff_sum += std::max(K - S_cur, 0.0);
			
		}
		
		return (payoff_sum / static_cast<double>(N)) * exp(-r * T);
		
	}
	
	
// Pricing a European vanilal call option using MC
// Create 3 seperate paths, each with either an increment , decrement or non-decrement
// based on delta_S, the stock path parameter
// void because have to return diferent prices so pass them as ref
void monte_carlo_call_price(const int num_sims, const double S, const double K, 
                            const double r, const double sig, const double T,
                            const double delta_S, double& price_Sp, double& price_S,
                            double& price_Sm){
		
								
								
		// Since we wish to use the same Gaussian random draws for each path, it 
		// is necessary to create 3 seperated adjusted stock paths for each 
		// increment/decrement of the asset delta_S
		double Sp_adjust = (S+delta_S) * exp(T*(r-0.5*sig*sig));
		double S_adjust = S * exp(T*(r-0.5*sig*sig));
		double Sm_adjust = (S-delta_S)* exp(T*(r-0.5*sig*sig));
		
		
		// The will store all the 'current' prices as the MC is carried out
		double Sp_cur = 0.0;
		double S_cur = 0.0;
		double Sm_cur = 0.0;
		
		
		// These are three separated pay-off sums for the final prices
		double payoff_sum_p = 0.0;
		double payoff_sum = 0.0;
		double payoff_sum_m = 0.0;
		
		
		// Loop over the number of similations
		for(int i = 0; i < num_sims; ++i) {
			
		   double gauss_bm = gaussian_box_muller(); // use the same gauss_bm for computing the 3 call prices
		
		
		// Adjust the three stock paths
		   double expgauss = exp(sig*sqrt(T)*gauss_bm);  // Precalculate
		   
		
	       Sp_cur = Sp_adjust * expgauss;
	       S_cur = S_adjust * expgauss;
	       Sm_cur = Sm_adjust * expgauss;
		
		
		// Calculate the continual payoff sum for each increment/decrement
	       payoff_sum_p += std::max(Sp_cur - K, 0.0);
	       payoff_sum += std::max(S_cur - K, 0.0);
           payoff_sum_m += std::max(Sm_cur - K , 0.0);
	
	}
	
	
	
	// There are three seperate prices
	    double df = exp(-r*T);
	    price_Sp = (payoff_sum_p / static_cast<double>(num_sims)) * df;
	    price_S = (payoff_sum / static_cast<double>(num_sims)) * df;
	    price_Sm = (payoff_sum_m / static_cast<double>(num_sims)) * df;
	
}



double compute_call_delta_mc(const int num_sims, const double S, const double K, const double r, 
                     const double sig, const double T, const double delta_S) {
					 // These values will be populated via the mc call price fct
					  
	double price_Sp = 0.0;
	double price_S = 0.0;
	double price_Sm = 0.0;
						 
						 
	// Call the MC prcer ofr each of the three stock paths 
	// We only need two for the delta (S and S + delta_S)
	monte_carlo_call_price(num_sims, S, K, r, sig, T, delta_S, price_Sp, price_S, price_Sm);
	
	return (price_Sp - price_S) / delta_S;
	
}


double compute_call_gamma_mc(const int num_sims, const double S, const double K, const double r,
                     const double sig, const double T, const double delta_S) {
						  
	double price_Sp = 0.0;
	double price_S = 0.0;
	double price_Sm = 0.0;
	
	// Calla the MC pricer fort each of the three paths 
	// (use the three in this case)
	
	monte_carlo_call_price(num_sims, S, K, r, sig, T, delta_S, price_Sp, price_S, price_Sm);
	
	return (price_Sp - 2*price_S + price_Sm) / (delta_S * delta_S);
	
}				
	
	
	
	
int main() {
	
	// First we initialize the parameters
	int N = 0;   
	double S = 0.0;
	double K = 0.0;
	double r = 0.0;
	double sig = 0.0;
	double T = 0.0;
	
	
	// Then we assign them
	cout << "Enter the number of simulations" << endl;
	cin >> N;
	
	cout << "Enter the number of initial stock price" << endl;
	cin >> S;
	
	cout << "Enter the strike price" << endl;
	cin >> K;
	
	cout << "Enter the risk-free interest rate" << endl;
	cin >> r;
	
	cout << "Enter the volatility" << endl;
	cin >> sig;
	
	cout << "Enter the maturity" << endl;
	cin >> T;
	
	// After we calculate the call and put prices via Monte Carlo
	double call_price = compute_call_price_mc(N, S, K, r, sig, T);
	double put_price = compute_put_price_mc(N, S, K, r, sig, T);
	
	double delta_S = 0.001;
	
	double call_delta = compute_call_delta_mc(N, S, K, r, sig, T, delta_S);
	double call_gamma = compute_call_gamma_mc(N, S, K, r, sig, T, delta_S);
	// European Put-Call parity
	double put_delta = call_delta - 1;
	// Option covexity
	double put_gamma = call_gamma;
	
	
	// Finally we output the parameters and prices	
	cout << "Number of paths: " << N << endl;
	cout << "Underlying price: " << S << endl;
	cout << "Strike price: " << K << endl;
	cout << "Risk-free rate: " << r << endl;
	cout << "Volatility sigma: " << sig << endl;
	cout << "Maturity: " << T << endl;
	
	cout << "Call price : " << call_price << endl;
	cout << "The delta for call:" << call_delta << endl;
	cout << "Put price : "<< put_price << endl;
	cout << "The delta for put:" << put_delta << endl;
	cout << "Gamma for both call & put:" << call_gamma << endl;
	
	
	return 0;
	
}
