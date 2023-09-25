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
 double monte_carlo_call_price(const int& N, const double& S,
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
 double monte_carlo_put_price(const int& N, const double& S, const double& K,
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
	double call = monte_carlo_call_price(N, S, K, r, sig, T);
	double put = monte_carlo_put_price(N, S, K, r, sig, T);
	
	// Finally we output the parameters and prices	
	cout << "Number of paths: " << N << endl;
	cout << "Underlying price: " << S << endl;
	cout << "Strike price: " << K << endl;
	cout << "Risk-free rate: " << r << endl;
	cout << "Volatility sigma: " << sig << endl;
	cout << "Maturity: " << T << endl;
	
	cout << "Call price : " << call << endl;
	cout << "Put price : "<< put << endl;
	
	return 0;
	
}
