functions {
  /**
 * Log probability of the intrinsic conditional autoregressive (ICAR) prior,
 * excluding additive constants. 
 *
 * @param phi Vector of parameters for spatial smoothing (on unit scale)
 * @param spatial_scale Scale parameter for the ICAR model
 * @param node1 
 * @param node2
 * @param k number of groups
 * @param group_size number of observational units in each group
 * @param group_idx index of observations in order of their group membership
 * @param has_theta If the model contains an independent partial pooling term, phi for singletons can be zeroed out; otherwise, they require a standard normal prior. Both BYM and BYM2 have theta.
 *
 * @return Log probability density of ICAR prior up to additive constant
 *
 * @author Connor Donegan
 *
 * @source
 *    Donegan, Connor. 2021. "Flexible functions for ICAR, BYM, and BYM2 models in Stan” Code repository. https://github.com/ConnorDonegan/Stan-IAR
 *  
 *    Morris, Mitzi, Katherine Wheeler-Martin, Dan Simpson, Stephen J Mooney, Andrew Gelman, and Charles DiMaggio. 2019. “Bayesian Hierarchical Spatial Models: Implementing the Besag York Mollié Model in Stan.” Spatial and Spatio-Temporal Epidemiology 31: 100301.
 *
 **/
real icar_normal_lpdf(vector phi, vector spatial_scale,
              int[] node1, int[] node2, 
              int k, int[] group_size, int[] group_idx,
              int has_theta) {
  real lp;
  int pos=1;
  lp = -0.5 * dot_self(phi[node1] - phi[node2]);
  if (has_theta) {
    for (j in 1:k) {
      /* sum to zero constraint for each connected group; singletons zero out */
      lp += normal_lpdf(sum(phi[segment(group_idx, pos, group_size[j])]) | 0, 0.001 * group_size[j]);
      pos += group_size[j];
    }
  } else {
    /* has no theta */
    for (j in 1:k) {
      if (group_size[j] > 1) {
    /* same as above for non-singletons: sum to zero constraint */
    lp += normal_lpdf(sum(phi[segment(group_idx, pos, group_size[j])]) | 0, 0.001 * group_size[j]);
      } else {
    /* its a singleton: independent std normal prior on phi */
    lp += normal_lpdf(phi[ segment(group_idx, pos, group_size[j]) ] | 0, spatial_scale[j]);
      }      
      pos += group_size[j];
    }
  }
  return lp;
}

/**
 * Combine local and global partial-pooling components into the convolved BYM term.
 *
 * @param phi local (spatially autocorrelated) component
 * @param theta global component
 * @param n number of spatial units
 * @param k number of connected groups
 * @param group_size number of observational units in each group
 * @param group_idx index of observations in order of their group membership
 *
 * @return BYM convolution vector
 *
 * @author Connor Donegan
 *
 * @source Donegan, Connor. 2021. "Flexible functions for ICAR, BYM, and BYM2 models in Stan” Code repository. https://github.com/ConnorDonegan/Stan-IAR
 */
vector convolve_bym(vector phi, vector theta,
		      int n, int k,
		      int[] group_size, int[] group_idx
		      ) {
  vector[n] convolution;
  int pos=1;
  for (j in 1:k) {
     if (group_size[j] == 1) {
        convolution[ segment(group_idx, pos, group_size[j]) ] = theta[ segment(group_idx, pos, group_size[j]) ];
    } else {
    convolution[ segment(group_idx, pos, group_size[j]) ] = phi[ segment(group_idx, pos, group_size[j]) ] + theta[ segment(group_idx, pos, group_size[j]) ];
  }
      pos += group_size[j];
  }
  return convolution;
}


/**
 * Create phi from phi_tilde, alpha_phi and spatial_scale: center and scale each graph component
 *
 * @param phi_tilde local component (spatially autocorrelated) 
 * @param phi_scale scale parameter for ICAR Normal model
 * @param inv_sqrt_scale_factor this scales the graph component, putting phi_tilde on standard normal scale; one scale per fully connected graph component.
 * @param n number of spatial units
 * @param k number of connected groups
 * @param group_size number of observational units in each group
 * @param group_idx index of observations in order of their group membership
 *
 * @return phi vector of spatially autocorrelated coefficients
 *
 * @author Connor Donegan
 *
 * @source Donegan, Connor. 2021. "Flexible functions for ICAR, BYM, and BYM2 models in Stan” Code repository. https://github.com/ConnorDonegan/Stan-IAR
 */
vector make_phi(vector phi_tilde, vector phi_scale,
		      vector inv_sqrt_scale_factor,
		      int n, int k,
		      int[] group_size, int[] group_idx
		      ) {
  vector[n] phi;
  int pos=1;
  for (j in 1:k) {
      phi[ segment(group_idx, pos, group_size[j]) ] = phi_scale[j] * inv_sqrt_scale_factor[j] * phi_tilde[ segment(group_idx, pos, group_size[j]) ];
    pos += group_size[j];
  }
  return phi;
}

 
/**
 * Log probability of the conditional autoregressive (CAR) model,
 * excluding additive constants.
 *
 * @param y Process to model
 * @param mu Mean vector
 * @param tau Scale parameter
 * @param rho Spatial dependence parameter
 * @param ImC Sparse representation of (I - C): non-zero values only
 * @param v Column indices for values in ImC
 * @param u Row starting indices for values in ImC
 * @param Cidx Indices for the off-diagonal elements in w
 * @param M_inv Diagonal elements from the inverse of the conditional  variances (M^-1 / tau^2)
 * @param lambda Eigenvalues of M^{-1/2}*C*M^{1/2}
 * @param n Length of y
 *
 * @return Log probability density of CAR prior up to additive constant
 *
 * @author Connor Donegan
*/
real car_normal_lpdf(vector y, vector mu,
		     real tau, real rho,
		     vector ImC, int[] v, int[] u, int[] Cidx,
		     vector M_inv, vector lambda,
		     int n) {
  vector[n] z = y - mu;  
  vector[num_elements(ImC)] ImrhoC = ImC; // (I - rho C)
  vector[n] zMinv = z .* M_inv;           // z' * M^-1
  vector[n] ImrhoCz;                      // (I - rho * C) * z
  vector[n] ldet_prec;
  ImrhoC[Cidx] = rho * ImC[Cidx];
  ImrhoCz = csr_matrix_times_vector(n, n, ImrhoC, v, u, z);
  for (i in 1:n) ldet_prec[i] = log1m(rho * lambda[i]);
  return 0.5 * (
		-2 * n * log(tau)
		+ sum(ldet_prec)
		- (1 / tau^2) * dot_product(zMinv, ImrhoCz)
		);
}

/**
 * Returns eigenvalues needed for the log determinant of
 * the CAR precision matrix and for identifying the
 * boundaries of permissible values of the spatial
 * autocorrelation parameter, rho.
 *
 * @param C connectivity matrix
 * @param M_diag constant entries from the diagonal
 *                of matrix M.
 *
 * @author Connor Donegan
*/
vector eMCM(matrix C, vector M_diag) {
  int n = num_elements(M_diag);
  vector[n] lambda;
  vector[n] invsqrtM;
  vector[n] sqrtM;
  for (i in 1:n) invsqrtM[i] = 1 / sqrt(M_diag[i]);
  for (i in 1:n) sqrtM[i] = sqrt(M_diag[i]);
  lambda = eigenvalues_sym(diag_matrix(invsqrtM) * C * diag_matrix(sqrtM));
  return (lambda);
}

}


data {
  int n;    // no. observations
  // BYM stuff  
  int<lower=1> k; // no. of groups
  int<lower=0> m; // no of components requiring additional intercepts
  matrix[n, m] A; // dummy variables for any extra graph component intercepts
  int group_size[k]; // observational units per group
  int group_idx[n]; // index of observations, ordered by group
  int<lower=1> n_edges; 
  int<lower=1, upper=n> node1[n_edges];
  int<lower=1, upper=n> node2[n_edges];
  int<lower=1, upper=k> comp_id[n];
  vector<lower=0>[k] inv_sqrt_scale_factor;
  // Mortality data
  int y[n];
  vector[n] log_at_risk;
  // ICE (covariate)
  vector[n] z;
  vector[n] z_se;
  // prior on Q.5/Q.1 (RR) -> prior on beta2
  real RII_prior[2];  // [mean, scale] Gaussian prior on RII induces prior on beta
  real IQR;           // Q.80 - Q.20 for the ICE (covariate), for prior on RII, beta
  matrix[n, n] C;    // row-standardized connectivity matrix: for for SLX
  // auto Gaussian model for ICE
  int nImC;           // size of w, which holds non-zero values of (I - C) "I minus C"
  int nC;             // Size of Cidx, to extract elements of C from w
  vector[nImC] w;
  int v[nImC];
  int u[n + 1];
  int Cidx[nC];
  vector[n] M_diag;
}

transformed data {
  // CAR 
  vector[n] mean_zero = rep_vector(0, n);
  vector[n] lambda = eMCM(C, M_diag);
  vector[n] M_inv = 1 ./ M_diag;
  // ICAR
  int<lower=0,upper=1> has_theta=1;
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
  real RII; // relative index of inequality: Q1 / Q5
  // BYM
  vector[n] phi_tilde;
  vector[n] theta_tilde;
  vector[m] alpha_phi;
  vector<lower=0>[k] spatial_scale; // separate scale per graph component (mainland, hawaii)
  real<lower=0> theta_scale;  
}

transformed parameters {
  real beta2 = log(RII^-1) / IQR;
  vector[n] phi = make_phi(phi_tilde, spatial_scale, inv_sqrt_scale_factor, n, k, group_size, group_idx);
  vector[n] theta = theta_tilde * theta_scale;
  vector[n] convolution = convolve_bym(phi, theta, n, k, group_size, group_idx);
  vector[n] wx = csr_matrix_times_vector(n, n, c_w, c_v, c_u, x);
  vector[n] y_mu = log_at_risk + alpha + A * alpha_phi + beta1 * wx + beta2 * x + convolution;  
}


model {
  // data model
  z ~ normal(x, z_se);
  x ~ car_normal(rep_vector(x_mu, n), x_tau, x_rho, w, v, u, Cidx, M_inv, lambda, n);
  x_mu ~ std_normal();
  x_tau ~ std_normal();
  // process model
  y ~ poisson_log(y_mu);
  // parameter model
  phi_tilde ~ icar_normal(spatial_scale, node1, node2, k, group_size, group_idx, has_theta);
  theta_tilde ~ std_normal();
  spatial_scale ~ std_normal();
  theta_scale ~ std_normal();
  alpha ~ normal(-5, 5);
  alpha_phi ~ normal(0, 5);
  beta1 ~ normal(0, 5);
  RII ~ normal(RII_prior[1], RII_prior[2]);
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

