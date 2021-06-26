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
 * @author Connor Donegan (Connor.Donegan@UTDallas.edu; Connor.Donegan@UTSouthwestern.edu)
 *
 * @source Donegan, Connor and Yongwan Chun and Daniel A. Griffith. 2021. Modeling community health with areal data: Bayesian inference with survey standard errors and spatial structure. Int. J. Env. Res. Public Health. 18, no. 13: 6856. https://doi.org/10.3390/ijerph18136856.
*/
real car_normal_lpdf(vector y, vector mu,
		     real tau, real rho,
		     vector ImC, int[] v, int[] u, int[] Cidx,
		     vector M_inv, vector lambda,
		     int n) {
  vector[n] z = y - mu;  
  vector[num_elements(ImC)] ImrhoC = ImC; // (I - C)
  vector[n] zMinv = z .* M_inv;           // z' * M^-1
  vector[n] ImrhoCz;                      // (I - rho * C) * z
  vector[n] ldet_prec;
  ImrhoC[Cidx] = rho * ImC[Cidx];         // (I - rho C)
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
