seed <- 1 # Simulation seed (1, ..., 10000) for reproducibility.
n <- 1000 #sample size 
tstar <- 10 # time horizon

# Install cmdstanr following the instructions at https://mc-stan.org/cmdstanr/
library(cmdstanr)

# On Windows, Rtools may be required to install CmdStan.
# Rtools45: https://cran.r-project.org/bin/windows/Rtools/rtools45/rtools.html

# Install CmdStan if no installation is configured or detected.
if (is.null(tryCatch(cmdstanr::cmdstan_path(), error = function(e) NULL))) {
  cmdstanr::install_cmdstan()
}

# ----------------------------------------------------------------------------
# Generate data
# ----------------------------------------------------------------------------

generate_data <- function(
    n = 1000,
    tstar = 10,
    seed = NULL,

    lambda_E0 = 0.18,
    beta_E0 = 0.20,
    lambda_E1 = 0.07,
    beta_E1 = -0.10,

    lambda_T0 = 0.020,
    beta_T0 = 0.10,
    lambda_T1 = 0.015,
    beta_T1 = 0.25
) {

  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Require an even sample size for equal treatment allocation.
  if (n %% 2 != 0) {
    stop("n must be even to assign equal numbers to treatment and control.")
  }

  # Generate the baseline covariate and assign treatment in a 1:1 ratio.
  X <- rnorm(n, mean = 0, sd = 1)

  Z <- as.integer(
    seq_len(n) %in% sample(seq_len(n), size = n / 2, replace = FALSE)
  )

  # Generate potential non-terminal event times.
  E0 <- rexp(
    n = n,
    rate = lambda_E0 * exp(beta_E0 * X)
  )

  E1 <- rexp(
    n = n,
    rate = lambda_E1 * exp(beta_E1 * X)
  )

  # Generate potential death times.
  T0 <- rexp(
    n = n,
    rate = lambda_T0 * exp(beta_T0 * X)
  )

  T1 <- rexp(
    n = n,
    rate = lambda_T1 * exp(beta_T1 * X)
  )

  # Select factual outcomes according to treatment assignment.
  E <- ifelse(Z == 1, E1, E0)
  T <- ifelse(Z == 1, T1, T0)

  # Calculate the observed time to the first event.
  Y <- pmin(E, T)

  # delta = 1: death precedes E; delta = 0: E precedes death.
  delta <- as.integer(T < E)

  observed_data <- data.frame(
    id = seq_len(n),
    X = X,
    Z = Z,
    Y = Y,
    delta = delta,
    T = T
  )

  list(
    observed_data = observed_data
  )
}

# ----------------------------------------------------------------------------
# Simulate data
# ----------------------------------------------------------------------------

sim_data <- generate_data(
  seed = seed
)

observed_data <- sim_data$observed_data

# ----------------------------------------------------------------------------
# Prepare Stan data
# ----------------------------------------------------------------------------

stan_data <- list(
  N = nrow(observed_data),
  X = as.vector(observed_data$X),
  Z = as.integer(observed_data$Z),
  Y = as.vector(observed_data$Y),
  delta = as.integer(observed_data$delta),
  T = as.vector(observed_data$T),
  tstar = tstar
)

# ----------------------------------------------------------------------------
# Specify initial values
# ----------------------------------------------------------------------------

init_fn <- function() {
  list(
    log_lambda_E0 = log(0.10),
    log_lambda_E1 = log(0.10),
    log_lambda_T0 = log(0.05),
    log_lambda_T1 = log(0.05),

    beta_E0 = 0.0,
    beta_E1 = 0.0,
    beta_T0 = 0.0,
    beta_T1 = 0.0
  )
}

# ----------------------------------------------------------------------------
# Define Stan model
# ----------------------------------------------------------------------------

stan_code <- "
data {
  int<lower=1> N;                        // Sample size
  vector[N] X;                          // Baseline covariate
  array[N] int<lower=0, upper=1> Z;      // Treatment assignment
  vector<lower=0>[N] Y;                 // Time to first event: min(E, T)
  array[N] int<lower=0, upper=1> delta;  // Indicator that death precedes E
  vector<lower=0>[N] T;                 // Observed factual death time
  real<lower=0> tstar;                    // Principal-stratum horizon
}

transformed data {
  // Possible always survivors have factual death times T(Z) > tstar.
  // Individuals with T(Z) <= tstar cannot be always survivors.
  int n_possible_AS = 0;
  array[N] int index_possible_AS;

  for (i in 1:N) {
    if (T[i] > tstar) {
      n_possible_AS += 1;
      index_possible_AS[n_possible_AS] = i;
    }
  }

  // Factual non-terminal event indicators among possible always survivors.
  vector[n_possible_AS] E0_lt_tstar_fixed =
    rep_vector(0.0, n_possible_AS);

  vector[n_possible_AS] E1_lt_tstar_fixed =
    rep_vector(0.0, n_possible_AS);

  if (n_possible_AS > 0) {
    for (j in 1:n_possible_AS) {
      int i = index_possible_AS[j];

      if (Z[i] == 0) {
        // E(0) is factual.
        if (delta[i] == 0) {
          // E(0) = Y is observed.
          E0_lt_tstar_fixed[j] = Y[i] < tstar ? 1.0 : 0.0;
        } else {
          // E(0) > T(0) = T > tstar.
          E0_lt_tstar_fixed[j] = 0.0;
        }
      } else {
        // E(1) is factual.
        if (delta[i] == 0) {
          // E(1) = Y is observed.
          E1_lt_tstar_fixed[j] = Y[i] < tstar ? 1.0 : 0.0;
        } else {
          // E(1) > T(1) = T > tstar.
          E1_lt_tstar_fixed[j] = 0.0;
        }
      }
    }
  }
}

parameters {
  // Log baseline hazards for potential non-terminal event times.
  real log_lambda_E0;
  real log_lambda_E1;

  // Log baseline hazards for potential death times.
  real log_lambda_T0;
  real log_lambda_T1;

  // Covariate coefficients.
  real beta_E0;
  real beta_E1;
  real beta_T0;
  real beta_T1;
}

model {
  // ----------------------------------------------------------------
  // Priors
  // ----------------------------------------------------------------

  log_lambda_E0 ~ normal(log(0.05), 1);
  log_lambda_E1 ~ normal(log(0.05), 1);
  log_lambda_T0 ~ normal(log(0.05), 1);
  log_lambda_T1 ~ normal(log(0.05), 1);

  beta_E0 ~ normal(0, 1);
  beta_E1 ~ normal(0, 1);
  beta_T0 ~ normal(0, 1);
  beta_T1 ~ normal(0, 1);

  // ----------------------------------------------------------------
  // Observed-data likelihood
  // ----------------------------------------------------------------

  for (i in 1:N) {
    real log_rate_E;
    real log_rate_T;

    if (Z[i] == 0) {
      log_rate_E = log_lambda_E0 + beta_E0 * X[i];
      log_rate_T = log_lambda_T0 + beta_T0 * X[i];
    } else {
      log_rate_E = log_lambda_E1 + beta_E1 * X[i];
      log_rate_T = log_lambda_T1 + beta_T1 * X[i];
    }

    // Factual death time T(Z) is observed for every individual.
    target += log_rate_T - T[i] * exp(log_rate_T);

    if (delta[i] == 0) {
      // The non-terminal event time E(Z) = Y is observed.
      target += log_rate_E - Y[i] * exp(log_rate_E);
    } else {
      // The non-terminal event time is right-censored at Y = T(Z).
      target += -Y[i] * exp(log_rate_E);
    }
  }
}

generated quantities {
  // ----------------------------------------------------------------
  // Finite-sample quantities
  // ----------------------------------------------------------------

  real n_AS_finite = 0.0;
  real numerator_finite = 0.0;
  real denominator_finite = 0.0;
  real psi_finite;

  // ----------------------------------------------------------------
  // Superpopulation quantities
  // ----------------------------------------------------------------

  vector<lower=0>[N] alpha_X;
  simplex[N] zeta_X;

  real numerator_super = 0.0;
  real denominator_super = 0.0;
  real psi_super;

  // ----------------------------------------------------------------
  // Finite-sample inference
  // ----------------------------------------------------------------

  // Impute missing potential outcomes from their posterior
  // predictive distributions.
  if (n_possible_AS > 0) {
    for (j in 1:n_possible_AS) {
      int i = index_possible_AS[j];

      real log_rate_E0;
      real log_rate_E1;
      real log_rate_T0;
      real log_rate_T1;

      real rate_E0;
      real rate_E1;
      real rate_T0;
      real rate_T1;

      log_rate_E0 = log_lambda_E0 + beta_E0 * X[i];
      log_rate_E1 = log_lambda_E1 + beta_E1 * X[i];
      log_rate_T0 = log_lambda_T0 + beta_T0 * X[i];
      log_rate_T1 = log_lambda_T1 + beta_T1 * X[i];

      rate_E0 = exp(log_rate_E0);
      rate_E1 = exp(log_rate_E1);
      rate_T0 = exp(log_rate_T0);
      rate_T1 = exp(log_rate_T1);

      if (Z[i] == 0) {
        // T(0) > tstar is factual; impute T(1).
        real T1_cf;

        T1_cf = exponential_rng(rate_T1);

        if (T1_cf > tstar) {
          real E1_cf;
          real E1_lt_tstar_cf;

          // Classify the individual as an always survivor.
          n_AS_finite += 1.0;

          // Use the factual E(0) indicator.
          denominator_finite += E0_lt_tstar_fixed[j];

          // Impute E(1).
          E1_cf = exponential_rng(rate_E1);
          E1_lt_tstar_cf = E1_cf < tstar ? 1.0 : 0.0;

          numerator_finite += E1_lt_tstar_cf;
        }

      } else {
        // T(1) > tstar is factual; impute T(0).
        real T0_cf;

        T0_cf = exponential_rng(rate_T0);

        if (T0_cf > tstar) {
          real E0_cf;
          real E0_lt_tstar_cf;

          // Classify the individual as an always survivor.
          n_AS_finite += 1.0;

          // Use the factual E(1) indicator.
          numerator_finite += E1_lt_tstar_fixed[j];

          // Impute E(0).
          E0_cf = exponential_rng(rate_E0);
          E0_lt_tstar_cf = E0_cf < tstar ? 1.0 : 0.0;

          denominator_finite += E0_lt_tstar_cf;
        }
      }
    }
  }

  if (n_AS_finite > 0.0 && denominator_finite > 0.0) {
    psi_finite = numerator_finite / denominator_finite;
  } else {
    psi_finite = not_a_number();
  }

  // ----------------------------------------------------------------
  // Superpopulation inference
  // ----------------------------------------------------------------

  // Draw Bayesian-bootstrap weights for F_X and compute psi_super
  // analytically, conditional on the posterior draw and these weights.
  alpha_X = rep_vector(1.0, N);
  zeta_X = dirichlet_rng(alpha_X);

  for (i in 1:N) {
    real rate_E0;
    real rate_E1;
    real rate_T0;
    real rate_T1;

    rate_E0 = exp(log_lambda_E0 + beta_E0 * X[i]);
    rate_E1 = exp(log_lambda_E1 + beta_E1 * X[i]);
    rate_T0 = exp(log_lambda_T0 + beta_T0 * X[i]);
    rate_T1 = exp(log_lambda_T1 + beta_T1 * X[i]);

    numerator_super += zeta_X[i] *
      (-expm1(-rate_E1 * tstar)) *
      exp(-tstar * (rate_T0 + rate_T1));

    denominator_super += zeta_X[i] *
      (-expm1(-rate_E0 * tstar)) *
      exp(-tstar * (rate_T0 + rate_T1));
  }

  psi_super = numerator_super / denominator_super;
}
"

# ----------------------------------------------------------------------------
# Write and compile Stan model
# ----------------------------------------------------------------------------

stan_file <- file.path(getwd(), "model.stan")

# Rewrite the file only if the Stan code has changed.
stan_lines <- strsplit(stan_code, "\n", fixed = TRUE)[[1]]

if (
  !file.exists(stan_file) ||
  !identical(readLines(stan_file, warn = FALSE), stan_lines)
) {
  writeLines(stan_lines, con = stan_file)
}

stan_model <- cmdstanr::cmdstan_model(
  stan_file = stan_file
)

# ----------------------------------------------------------------------------
# Fit Stan model
# ----------------------------------------------------------------------------

fit <- stan_model$sample(
  data = stan_data,
  init = init_fn,

  chains = 1,

  iter_warmup = 5000,
  iter_sampling = 5000,

  adapt_delta = 0.9,
  max_treedepth = 12,

  seed = seed,
  refresh = 50
)

# ----------------------------------------------------------------------------
# Extract posterior draws
# ----------------------------------------------------------------------------

variables_to_save <- c(
  "psi_finite",
  "psi_super"
)

draws <- fit$draws(
  variables = variables_to_save,
  format = "draws_df"
)

CI_finite <- as.numeric(
  quantile(draws$psi_finite, c(0.025, 0.975), na.rm = TRUE)
)

CI_super <- as.numeric(
  quantile(draws$psi_super, c(0.025, 0.975), na.rm = TRUE)
)

cat(sprintf(
  paste0(
    "95%% credible intervals:\n",
    "  Finite-sample estimand:    [%.3f, %.3f]\n",
    "  Super-population estimand: [%.3f, %.3f]\n"
  ),
  CI_finite[1], CI_finite[2],
  CI_super[1], CI_super[2]
))