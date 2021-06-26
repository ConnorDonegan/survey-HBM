functions {
#include CAR-functions.stan
}

data {
  // Mortality data
  int n;    // no. observations
  int y[n];
  vector[n] log_at_risk;
  // ICE (covariate)
  vector[n] z;
  vector[n] z_se;
  // prior on Q.5/Q.1 (RR) -> prior on beta2
  real RII_prior[2];  // [mean, scale] Gaussian prior on RII induces prior on beta
  real IQR;           // Q.80 - Q.20 for the ICE (covariate), for prior on RII, beta
  matrix[n, n] C;    // row-standardized connectivity matrix: for SLX
  // data for auto-Gaussian models
  int nImC;           // size of w, which holds non-zero values of (I - C) "I minus C"
  int nC;             // Size of Cidx, to extract elements of C from w
  vector[nImC] w;
  int v[nImC];
  int u[n + 1];
  int Cidx[nC];
  vector[n] M_diag;
}

transformed data {
  // CAR stuff
  vector[n] lambda = eMCM(C, M_diag);
  vector[n] M_inv = 1 ./ M_diag;
  // sparse representation of row-standardized connectivity matrix
  vector[nC] c_w = csr_extract_w(C);                
  int c_v[nC] = csr_extract_v(C);
  int c_u[n + 1] = csr_extract_u(C);  
}

parameters {
  // data model
  vector[n] x;  
  real x_mu;
  real<lower=0> x_tau;
  real<lower=1/min(lambda), upper=1/max(lambda)> x_rho;
  // process mean plus coefficients for W*ICE and ICE
  real alpha;
  real beta1;
  real RII; // relative index of inequality: P80 / P20
  // spatial: outcome
  vector[n] phi;
  real<lower=0> phi_tau;
  real<lower=1/min(lambda), upper=1/max(lambda)> phi_rho;
}

transformed parameters {
  real beta2 = log(RII^-1) / IQR;
  vector[n] wx = csr_matrix_times_vector(n, n, c_w, c_v, c_u, x);
  vector[n] y_mu = log_at_risk + phi; 
  vector[n] phi_mu = alpha + beta1 * wx + beta2 * x;
}


model {
  // data model
  vector[n] x_mu_vec = rep_vector(x_mu, n);
  z ~ normal(x, z_se);
  x ~ car_normal(x_mu_vec, x_tau, x_rho, w, v, u, Cidx, M_inv, lambda, n);
  x_mu ~ std_normal();
  x_tau ~ std_normal();
  // process models
  y ~ poisson_log(y_mu);
  phi ~ car_normal(phi_mu, phi_tau, phi_rho, w, v, u, Cidx, M_inv, lambda, n);
  // parameter models
  alpha ~ normal(-5, 5);
  beta1 ~ normal(0, 5);
  RII ~ normal(RII_prior[1], RII_prior[2]);
  phi_tau ~ std_normal();
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

