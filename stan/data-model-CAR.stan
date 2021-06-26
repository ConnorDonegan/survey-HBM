functions {
#include CAR-functions.stan
}

data {
  int n;
  vector[n] z;
  vector[n] z_se;
  // priors
  vector[3] prior_mu;
  vector[3] prior_tau;
  // CAR data
  matrix[n, n] C;
  int nImC;
  int nC;
  vector[nImC] w;
  int v[nImC];
  int u[n + 1];
  int Cidx[nC];
  vector[n] M_diag;
}

transformed data {
  vector[n] lambda = eMCM(C, M_diag);
  vector[n] M_inv = 1 ./ M_diag;  
}

parameters {
  vector[n] x;  
  real mu;
  real<lower=0> tau;
  real<lower=1/min(lambda), upper=1/max(lambda)> rho;
}

transformed parameters {
}

model {
  z ~ normal(x, z_se);
  x ~ car_normal(rep_vector(mu, n), tau, rho, w, v, u, Cidx, M_inv, lambda, n);
  mu ~ student_t(prior_mu[1], prior_mu[2], prior_mu[3]);
  tau ~ student_t(prior_tau[1], prior_tau[2], prior_tau[3]);
 }

generated quantities {
  vector[n] delta = x - z;
  vector[n] trend = rho * C * (z - mu);
}
