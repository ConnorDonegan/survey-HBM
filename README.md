

-   [A model for small-area survey
    data](#a-model-for-small-area-survey-data)
-   [CAR models in Stan](#car-models-in-stan)
-   [Working with the US county mortality
    data](#working-with-the-us-county-mortality-data)
-   [CAR model for all-cause mortality
    rates](#car-model-for-all-cause-mortality-rates)
-   [All-cause mortality and the Index of Concentration at the Extremes
    (ICE)](#all-cause-mortality-and-the-index-of-concentration-at-the-extremes-ice)

This repository contains code and data for the paper:

> Donegan, Connor, Yongwan Chun and Daniel A. Griffith. 2021. **Modeling
> Community Health with Areal Data: Bayesian Inference with Survey
> Standard Errors and Spatial Structure** *International Journal of
> Environmental Research and Public Health* 18, no. 13: 6856.
> <https://doi.org/10.3390/ijerph18136856>.

The purpose of the repository is to demonstrate implementation of the
methodology and to share the underlying R and Stan code. For discussion
and details, see the published article. For the R scripts used to create
the figures and results reported in the paper, see the online
supplementary material published with the article.

### A model for small-area survey data

This section provides Stan code for implementing the models for small
area survey data, as presented in Section 3 of the article.

To demonstrate the methodology, we model American Community Survey
estimates of health insurance coverage rates for Milwaukee County census
tracts. This code chunk loads the file `tract-data.rds`, a simple
features (`sf`) object. (The data, without geometry, is also stored as
`acs-data.csv`.)

``` r
pkgs <- c("tidyverse", "spdep", "sf", "rstan")
silent=lapply(pkgs, require, character.only = TRUE)
rstan_options(auto_write = TRUE, javascript = FALSE)

## source some functions
source("r-functions.R")

## compile the Stan model
car.hbm <- stan_model("stan/data-model-CAR.stan")

## prepare data to pass to Stan:
# read in the data
df = read_rds("data/tract-data.rds")
# create a binary spatial adjacency matrix
A <- spdep::nb2mat(spdep::poly2nb(df), style = "B")
# convert it to a list with all the information required for our CAR model
dl <- prep_car_data(A)
# add survey estimates and standard errors to the list
dl$z <- df$insurance
dl$z_se <- df$insurance.se
dl$n <- nrow(df)
## specify priors based on our vague prior information (degrees of freedom, location, and scale of a Student's t distribution)
dl$prior_mu <- c(10, 80, 20)
dl$prior_tau <- c(10, 20, 20)

## draw samples from the posterior distribution of parameters (requires about 10 seconds)
S <- sampling(car.hbm, data = dl, chains = 4, cores = 4)
```

The following code chunk prints convergence diagnostics (split Rhat) and
effective sample sizes. The effective sample size (ESS) is relevant,
nominal sample size (i.e., actual number of samples drawn from the
posterior distribution) much less so. Increasing the ESS reduces the
MCMC standard error. Given the efficiency of the model (high ESS), we
could reduce sampling time by cutting down on the number of samples
collected, if needed:

``` r
## check MCMC diagnostics (Hamiltonian Monte Carlo)
check_hmc_diagnostics(S)
```

    ## 
    ## Divergences:

    ## 0 of 4000 iterations ended with a divergence.

    ## 
    ## Tree depth:

    ## 0 of 4000 iterations saturated the maximum tree depth of 10.

    ## 
    ## Energy:

    ## E-BFMI indicated no pathological behavior.

``` r
## you can print quantiles, or plot results
## stan_rhat(S)
## stan_ess(S)
quantile( apply(as.matrix(S), 2, Rhat) )
```

    ##        0%       25%       50%       75%      100% 
    ## 0.9997505 0.9999724 1.0005311 1.0006829 1.0060352

``` r
quantile( apply(as.matrix(S), 2, ess_bulk) )
```

    ##       0%      25%      50%      75%     100% 
    ## 1048.434 1974.684 7074.647 7952.598 9863.585

``` r
quantile( apply(as.matrix(S), 2, ess_tail) )
```

    ##        0%       25%       50%       75%      100% 
    ##  900.9604  901.8681 2698.9552 2948.8143 3476.0234

The `plot.res` function is stored in the `r-functions.R` file, and it
provides the visual diagnostics discussed in the paper:

``` r
plot.res(S, z = dl$z, sf = df, W = dl$C)
```

<img src="README_files/figure-markdown_github/unnamed-chunk-3-1.png" style="display: block; margin: auto;" />

The following code reports, first, the variance of the raw ACS estimates
for insurance coverage in Milwaukee County census tracts, and then, the
probability distribution for the actual variance of insurance coverage.
This shows, again, that the variance declines from 32 to (a posterior
mean/median of) 21:

``` r
var(dl$z)
```

    ## [1] 32.14744

``` r
x <- as.matrix(S, pars = "x")
quantile( apply(x, 1, var  ) )
```

    ##       0%      25%      50%      75%     100% 
    ## 14.77782 19.73126 20.86281 22.09522 27.62099

### CAR models in Stan

Stan does not have an efficient auto-Gaussian model specification
\`built in’. We wrote the following implementation for the log-pdf of
the auto-Gaussian model:

``` stan
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
 * @param M_inv Diagonal elements from the inverse of the conditional  variances (M^-1), constant terms only
 * @param lambda Eigenvalues of M^{-1/2}*C*M^{1/2}
 * @param n Length of y
 *
 * @return Log probability density of CAR model up to additive constant
 *
 * @author Connor Donegan Connor.Donegan@UTDallas.edu
 *
 *@source 
 *
 * Donegan, Connor and Yongwan Chun and Daniel A. Griffith. 2021. "Modeling Community Health with Areal Data: Bayesian Inference with Survey Standard Errors and Spatial Structure." International Journal of Environmental Research and Public Health 18, no. 13: 6856. https://doi.org/10.3390/ijerph18136856.
 *
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
```

The code itself may be presented in a separate paper, hopefully in the
near future. A benefit of this code is that it permits any valid CAR
model specification (e.g., distance weighted connectivity structures)
and it is computationally efficient.

The following R function prepares the input data for the CAR model. It
requires a binary spatial connectivity matrix as its argument:

``` r
#' Prepare data for Stan CAR model
#'
#' @param A  Binary adjacency matrix
#' @param lambda  If TRUE, return eigenvalues required for calculating the log determinant
#' of the precision matrix and for determining the range of permissible values of rho.
#' @param cmat  Return the full matrix C if TRUE.
#' 
#' @details
#'
#' The CAR model is Gauss(Mu, Sigma), Sigma = (I - rho C)^{-1} M.
#' This function implements the specification of C and M known as the
#' "neighborhoods" or "weighted" (WCAR) specification (see Cressie and Wikle 2011,
#' pp. 186-88, for CAR specifications).
#'
#' @source
#'
#'  Cressie, Noel and Christopher K. Wikle. Statistics for Spatio-Temporal Data. Wiley.
#'
#'  Donegan, Connor and Yongwan Chun and Daniel A. Griffith. 2021. "Modeling Community Health with Areal Data: Bayesian Inference with Survey Standard Errors and Spatial Structure." International Journal of Environmental Research and Public Health 18, no. 13: 6856. https://doi.org/10.3390/ijerph18136856.
#'
#' @return A list containing all of the data elements required by the Stan CAR model.
#'
#' @author Connor Donegan (Connor.Donegan@UTDallas.edu)
#' 
prep_car_data <- function(A, lambda = FALSE, cmat = TRUE) {
    n <- nrow(A)    
    Ni <- rowSums(A)
    C <- A / Ni
    M_diag <- 1 / Ni
    stopifnot( isSymmetric.matrix(C %*% diag(M_diag), check.attributes = FALSE) )
    car.dl <- rstan::extract_sparse_parts(diag(n) - C)
    car.dl$Cidx <- which( car.dl$w != 1 )
    car.dl$nImC <- length(car.dl$w)
    car.dl$nC <- length(car.dl$Cidx)
    car.dl$M_diag <- M_diag
    car.dl$n <- n
    if (lambda) {
        MCM <- diag( 1 / sqrt(M_diag) ) %*% C %*% diag( sqrt(M_diag) )
        lambda <- eigen(MCM)$values
        cat ("Range of permissible rho values: ", 1 / range(lambda), "\n")
        car.dl$lambda <- lambda
    }
    if (cmat) car.dl$C <- C
    return (car.dl)
}
```

The full Stan model for areal survey data is print below, and is also
stored in the file `stan/car-data-model.stan`. The CAR pdf is read in
from its own file, `CAR-functions.stan` (it must be stored in the same
directory as `car-data-model.stan`).

``` stan
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
```

### Working with the US county mortality data

The following code chunk shows how to read in the county mortality data
and the connectivity structure. This is written to have no dependence on
external packages. The male mortality data may be loaded by simply
replacing all instaces of \`\`female’’ with ‘’male’’ in the following
code chunk.

``` r
## read in a data.frame with all the mortality and county ACS data
df = read.csv("data/county-data.csv")

## read in the connectivity structure
A <- read.csv("data/county-connectivity-female-df.csv")
## GEOIDs are in the column names
x = names(A)
## drop the leading character (read.csv prepends an X to the column names)
geos <- as.numeric( gsub("X", "", x) )
## convert A from data.frame to matrix
A <- as.matrix(A)

## this code ensures that the connectivity matrix is aligned with the mortality data
   ## this is equilivalent to: drop Alaska, drop missing values of deaths.female, drop missing values of ICE.
female.df <- df[ which( df$GEOID %in% geos ) , ]
## check that the order is correct
all( female.df$GEOID == geos )
```

    ## [1] TRUE

``` r
## GEOIDs are all five digits, zero-padded on the left when needed
female.df$GEOID <- str_pad(female.df$GEOID, width = 5, pad = "0", side = "left")
## double check this all aligns correctly: calculate Moran coefficient for log-mortality rates
mc( log(female.df$rate.female), A )
```

    ## [1] 0.568

### CAR model for all-cause mortality rates

The paper uses an intrinsic conditional autoregressive (ICAR) model to
capture spatial autocorrelation in the mortality rates. This code shows
how to use the CAR model for the mortality rates. This is actually much
more efficient than the ICAR model, and does not require any adjustments
for the disconnected graph structure.

Below is a Stan model for the mortality rates (without covariates).
Notice that the mean log-mortality rate, `alpha`, enters into the mean
of the CAR model. When covariates are added to the model, they should
also be added in this manner. Specifically, if we have a covariate, `x`,
and coefficient, `beta`, then the mean of the CAR model would become:
`vector[n] phi_mu = alpha + x * beta;`

``` stan
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
```

``` r
## compile the Stan model
CAR_pois <- stan_model("stan/CAR-poisson.stan")
## prepare the data for Stan
car_dl <- prep_car_data(A, lambda = FALSE)
## add outcome data to the list
car_dl$y <- female.df$deaths.female
car_dl$log_at_risk <- log( female.df$pop.at.risk.female )
car_dl$n <- nrow(female.df)
## draw samples from the joint posterior distribution of parameters
S <- sampling(CAR_pois, data = car_dl, chains = 4, cores = 4, iter = 800, refresh = 10)
```

The log-mortality rates are stored in the parameter `phi`. Here we
calculate and map the mean of the posterior probability distributions
for each county mortality rate:

``` r
## county mortality rates per 100,000
phi <- as.matrix(S, pars = "phi")
phi <- exp( phi ) * 100e3
female.df$phi <- apply(phi, 2, mean)

## join results to a simple features spatial data frame
sp.df <- readr::read_rds("data/county-data.rds")
sp.df <- left_join(sp.df, female.df, by = "GEOID")

## map mortality rates
sp.df %>%
  ggplot() +
  geom_sf(aes(fill = phi),
          lwd = 0.01
          ) +
  scale_fill_gradient(
    low = "wheat2",
    high= "black", 
#    midpoint = alpha,
#    mid = "white",
    name = "Female deaths\n per 100,000,\n ages 55-64",
    na.value = "grey90"
  ) +
  theme_void() +
  theme(     
       legend.position = "bottom",
       legend.key.width = unit(2, 'cm')
       )  
```

![](README_files/figure-markdown_github/unnamed-chunk-11-1.png)

### All-cause mortality and the Index of Concentration at the Extremes (ICE)

This section shows how to fit the female all-cause mortality model from
the paper, but again using the CAR model instead of the ICAR (BYM)
model.

We will re-use the same list of data for this model as for the previous
CAR model, but we add the covariate data, covariate standard errors, and
some prior parameters:

``` r
## ICE: index of concentration at the extremes
car_dl$z <- female.df$ICE - mean(female.df$ICE)
## ICE standard errors
car_dl$z_se <- female.df$ICE.se
## these are for the prior distribution, see the paper for details
car_dl$IQR <- 0.179
car_dl$RII_prior <- c(1.6, 0.3)
```

Compile the Stan model and then draw samples from the posterior
distribution of parameters:

``` r
## compile model
CAR_hbm <- stan_model("stan/CAR-HBM.stan")
## draw samples
S <- sampling(CAR_hbm, data = car_dl, chains = 4, cores = 4, iter = 800, refresh = 10)
```

The following code produces a summary of the parameter values and
quantities of interest, including the mean county mortality rate and a
measure of inequality (the ratio of the 90th to 10th percentile of
counties ordered by their mortality rates):

``` r
## view some results 
print(S, pars = c("alpha", "beta1", "beta2", "phi_tau", "phi_rho"))
```

    ## Inference for Stan model: CAR-HBM.
    ## 4 chains, each with iter=800; warmup=400; thin=1; 
    ## post-warmup draws per chain=400, total post-warmup draws=1600.
    ## 
    ##          mean se_mean   sd  2.5%   25%   50%   75% 97.5% n_eff Rhat
    ## alpha   -4.86       0 0.05 -4.97 -4.89 -4.86 -4.83 -4.75  1411 1.00
    ## beta1    0.31       0 0.07  0.18  0.27  0.32  0.36  0.45  1637 1.00
    ## beta2   -1.67       0 0.03 -1.73 -1.69 -1.67 -1.65 -1.60  1641 1.00
    ## phi_tau  0.23       0 0.01  0.22  0.23  0.23  0.24  0.24   489 1.01
    ## phi_rho  1.00       0 0.00  1.00  1.00  1.00  1.00  1.00  1771 1.00
    ## 
    ## Samples were drawn using NUTS(diag_e) at Tue Jun 29 22:53:41 2021.
    ## For each parameter, n_eff is a crude measure of effective sample size,
    ## and Rhat is the potential scale reduction factor on split chains (at 
    ## convergence, Rhat=1).

``` r
# 90% cred. interval for the mean county mortality rate per 100,000 
alpha <- exp( as.matrix(S, pars = "alpha") )
quantile(alpha, probs = c(0.05, 0.95)) * 100e3
```

    ##       5%      95% 
    ## 707.7380 841.0667

``` r
# inequality in county mortality rates per 100,000
phi <- as.matrix(S, pars = "phi")
phi <- exp(phi)
Q10 <- apply(phi, 1, quantile, probs = 0.1)
Q90 <- apply(phi, 1, quantile, probs = 0.90)
## the 10th percentile:
mean(Q10) * 100e3
```

    ## [1] 529.0343

``` r
## the 90th percentile:
mean(Q90) * 100e3
```

    ## [1] 1117.06

``` r
## Relative index of inequality: p90/p10:
mean( Q90 / Q10 )
```

    ## [1] 2.11159
