functions {
#include CAR-functions.stan
}

data {
 // mortality data
  int n;
  int y[n];
  vector[n] log_at_risk;
  // CAR data
  int nImC;
  int nC;
  vector[nImC] w;
  int v[nImC];
  int u[n + 1];
  int Cidx[nC];
  vector[n] M_diag;
  // connectivity matrix (C), or eigenvalues (lambda)
  matrix[n, n] C;
//  vector[n] lambda;
}

transformed data {
  // calculate eigenvalues if not provided as data
  vector[n] lambda = eMCM(C, M_diag);
  vector[n] M_inv = 1 ./ M_diag;  
}

parameters {
  // mean county-level log-mortality rate
  real alpha;
  // log-mortality rates
  vector[n] phi;
  // CAR parameters for phi
  real<lower=0> tau;
  real<lower=1/min(lambda), upper=1/max(lambda)> rho;
}

transformed parameters {
  vector[n] y_mu = log_at_risk + phi; 
}

model {
  vector[n] phi_mu = rep_vector(alpha, n);
  y ~ poisson_log(y_mu);
  phi ~ car_normal(phi_mu, tau, rho, w, v, u, Cidx, M_inv, lambda, n);
  // prior distributions on alpha and tau
    // (this is not a 'default' prior for alpha---these should be based on your subject matter)
  alpha ~ normal(-5, 5);
  tau ~ std_normal();
 }

generated quantities {
  vector[n] log_lik;
  vector[n] yrep;
  vector[n] residual;
  vector[n] fitted;
  for (i in 1:n) {
    fitted[i] = exp(y_mu[i]);
    residual[i] = fitted[i] - y[i];
    log_lik[i] = poisson_log_lpmf(y[i] | y_mu[i]);
    if (y_mu[i] > 20) {
      print("f[i] too large (>20) for poisson_log_rng");
      yrep[i] = -1;
    } else {
    yrep[i] = poisson_log_rng(y_mu[i]);
    }
  }
}

