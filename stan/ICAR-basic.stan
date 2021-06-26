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
  // prior on Q.5/Q.1 (RR) -> prior on beta2
  real RII_prior[2];  // [mean, scale] Gaussian prior on RII induces prior on beta
  real IQR;           // Q.80 - Q.20 for the ICE (covariate), for prior on RII, beta
  matrix[n, n] C;     // row-standardized connectivity matrix: for for SLX
  int nC;             // for sparse representation of C
}

transformed data {
  // ICAR
  int<lower=0,upper=1> has_theta=1;
  // sparse representation of row-standardized connectivity matrix
  vector[nC] c_w = csr_extract_w(C);                
  int c_v[nC] = csr_extract_v(C);
  int c_u[n + 1] = csr_extract_u(C);  
}

parameters {
  // process mean plus coefficients for W*ICE and ICE
  real alpha;
  real beta1;
  real RII; // relative index of inequality: Q1 / Q5
  // BYM
  vector[n] phi_tilde;
  vector[n] theta_tilde;
  vector[m] alpha_phi;
  vector<lower=0>[k] spatial_scale; // separate scale per graph component (mainland, hawaii, Alaska)
  real<lower=0> theta_scale;  
}

transformed parameters {
  real beta2 = log(RII^-1) / IQR;
  vector[n] phi = make_phi(phi_tilde, spatial_scale, inv_sqrt_scale_factor, n, k, group_size, group_idx);
  vector[n] theta = theta_tilde * theta_scale;
  vector[n] convolution = convolve_bym(phi, theta, n, k, group_size, group_idx);
  vector[n] wz = csr_matrix_times_vector(n, n, c_w, c_v, c_u, z);
  vector[n] y_mu = log_at_risk + alpha + A * alpha_phi + beta1 * wz + beta2 * z + convolution;  
}


model {
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

