 

monitor_levels <- c(
  200,
  100,
  50,
  20,
  10
)

results_sparse <- data.frame()

for(m in monitor_levels){
  
  degradation <- sqrt(200/m)
  
  T_sparse <- T_pinn +
    matrix(
      rnorm(
        length(T_pinn),
        0,
        0.01 * degradation
      ),
      nrow(T_pinn),
      ncol(T_pinn)
    )
  
  operator_corr <- cor(
    as.vector(T_true),
    as.vector(T_sparse)
  )
  
  n <- nrow(T_true)
  
  A <- rbinom(n,1,0.3)
  
  E_true <- T_true %*% A
  E_est <- T_sparse %*% A
  
  beta0 <- 1
  beta1 <- 2
  beta2 <- 3
  
  Y <- beta0 +
    beta1*A +
    beta2*E_true +
    rnorm(n,0,0.5)
  
  fit <- lm(
    Y ~ A + E_est
  )
  
  beta_hat <- coef(fit)[3]
  
  results_sparse <- rbind(
    results_sparse,
    data.frame(
      monitors = m,
      operator_corr = operator_corr,
      spillover_bias = beta_hat - beta2
    )
  )
}

print(results_sparse)

write.csv(
  results_sparse,
  "results_sparse.csv",
  row.names = FALSE
)
 
