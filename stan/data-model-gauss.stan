
data {
  int n;
  vector[n] z;
  vector[n] z_se;
}

parameters {
  vector[n] x;  
  real mu;
  real<lower=0> tau;
}

model {
  z ~ normal(x, z_se);
  x ~ normal(mu, tau);
  mu ~ student_t(10, 80, 20);
  tau ~ student_t(10, 15, 10);
 }

generated quantities {
  vector[n] delta = x - z;
}
