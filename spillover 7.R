# =====================================================
# STEP 0
# SAVE ALL SIGMA DATASETS
# =====================================================

all_sigma_data <- list()

# =====================================================
# STEP 1
# SIGMA GRID SEARCH
# =====================================================

sigma_grid <- c(
  25,
  50,
  75,
  100,
  150,
  200,
  300
)

sigma_results <- list()

# =====================================================
# STEP 2
# LOOP OVER SIGMAS
# =====================================================

for(sig in sigma_grid){
  
  cat(
    "Sigma:",
    sig,
    "\n"
  )
  
  # --------------------------------------------------
  # GAUSSIAN KERNEL
  # --------------------------------------------------
  
  gaussian_kernel <- exp(
    
    -(distance_matrix^2) /
      
      (2 * sig^2)
  )
  
  diag(gaussian_kernel) <- 0
  
  # --------------------------------------------------
  # SPILLOVER STORAGE
  # --------------------------------------------------
  
  spillover_results <- list()
  
  # ==================================================
  # LOOP OVER DATES
  # ==================================================
  
  for(d in unique_dates){
    
    # ----------------------------------------------
    # SINGLE DAY DATA
    # ----------------------------------------------
    
    day_data <- panel_data %>%
      
      filter(
        date_local == d
      )
    
    # ----------------------------------------------
    # COMPLETE COUNTY SET
    # ----------------------------------------------
    
    day_data <- tibble(
      
      county_name = county_levels
      
    ) %>%
      
      left_join(
        
        day_data,
        
        by = "county_name"
      )
    
    # ----------------------------------------------
    # FIX ORDER
    # ----------------------------------------------
    
    day_data$county_name <- factor(
      
      day_data$county_name,
      
      levels = county_levels
    )
    
    day_data <- day_data %>%
      
      arrange(county_name)
    
    # ----------------------------------------------
    # PM VECTOR
    # ----------------------------------------------
    
    pm_vector <- day_data$pm25
    
    # ----------------------------------------------
    # VALID OBSERVATIONS
    # ----------------------------------------------
    
    valid_idx <- !is.na(pm_vector)
    
    # ----------------------------------------------
    # SUBSET KERNEL
    # ----------------------------------------------
    
    sub_kernel <- gaussian_kernel[
      ,
      valid_idx
    ]
    
    sub_pm <- pm_vector[
      valid_idx
    ]
    
    # ----------------------------------------------
    # ROW NORMALIZATION
    # ----------------------------------------------
    
    row_weights <- rowSums(
      sub_kernel
    )
    
    row_weights[
      row_weights == 0
    ] <- NA
    
    # ----------------------------------------------
    # SPILLOVER
    # ----------------------------------------------
    
    spillover <- (
      
      sub_kernel %*%
        sub_pm
      
    ) / row_weights
    
    # ----------------------------------------------
    # SAVE SPILLOVER
    # ----------------------------------------------
    
    day_data$spillover <- as.numeric(
      spillover
    )
    
    spillover_results[[as.character(d)]] <- day_data
    
  }   # END DATE LOOP
  
  # ==================================================
  # COMBINE ALL DAYS
  # ==================================================
  
  sigma_data <- bind_rows(
    spillover_results
  )
  
  # ==================================================
  # FILTER MODEL DATA
  # ==================================================
  
  sigma_data <- sigma_data %>%
    
    filter(
      
      !is.na(pm25),
      
      !is.na(pm25_lag1),
      
      !is.na(pm25_lag2),
      
      !is.na(wind_speed),
      
      !is.na(temperature),
      
      !is.na(spillover)
    )
  
  # ==================================================
  # SAVE SIGMA DATASET
  # ==================================================
  
  all_sigma_data[[as.character(sig)]] <- sigma_data
  
  # ==================================================
  # MODEL
  # ==================================================
  
  fit <- lm(
    
    pm25 ~
      
      pm25_lag1 +
      
      pm25_lag2 +
      
      spillover +
      
      wind_speed +
      
      temperature,
    
    data = sigma_data
  )
  
  # ==================================================
  # PREDICTIONS
  # ==================================================
  
  pred <- predict(fit)
  
  # ==================================================
  # SAVE RESULTS
  # ==================================================
  
  sigma_results[[as.character(sig)]] <- tibble(
    
    sigma = sig,
    
    AIC = AIC(fit),
    
    RMSE = rmse(
      sigma_data$pm25,
      pred
    ),
    
    R2 = summary(fit)$r.squared,
    
    spillover_beta =
      coef(fit)["spillover"]
  )
  
}   # END SIGMA LOOP

# =====================================================
# STEP 3
# FINAL SIGMA RESULTS
# =====================================================

sigma_results <- bind_rows(
  sigma_results
)

print(sigma_results)

# =====================================================
# STEP 4
# BEST SIGMA
# =====================================================

best_sigma <- sigma_results %>%
  
  arrange(AIC) %>%
  
  slice(1)

print(best_sigma)

# =====================================================
# STEP 5
# BEST SIGMA VALUE
# =====================================================

best_sigma_value <- best_sigma$sigma

print(best_sigma_value)

# =====================================================
# STEP 6
# EXTRACT BEST DATASET
# =====================================================

model_data <- all_sigma_data[[as.character(best_sigma_value)]]

# =====================================================
# STEP 7
# CHECK
# =====================================================

cat(
  "model_data rows:",
  nrow(model_data),
  "\n"
)

head(model_data)

# =====================================================
# STEP 8
# CORRELATION MATRIX
# =====================================================

cor_matrix <- model_data %>%
  
  select(
    
    pm25,
    
    pm25_lag1,
    
    pm25_lag2,
    
    spillover,
    
    wind_speed,
    
    temperature
  ) %>%
  
  cor(
    use = "complete.obs"
  )

print(cor_matrix)

# =====================================================
# STEP 9
# BASELINE MODEL
# =====================================================

baseline_model <- lm(
  
  pm25 ~
    
    pm25_lag1 +
    
    pm25_lag2 +
    
    wind_speed +
    
    temperature,
  
  data = model_data
)

summary(baseline_model)

# =====================================================
# STEP 10
# SPILLOVER MODEL
# =====================================================

spillover_model <- lm(
  
  pm25 ~
    
    pm25_lag1 +
    
    pm25_lag2 +
    
    spillover +
    
    wind_speed +
    
    temperature,
  
  data = model_data
)

summary(spillover_model)

# =====================================================
# STEP 11
# MODEL COMPARISON
# =====================================================

AIC(
  baseline_model,
  spillover_model
)

# =====================================================
# STEP 12
# VIF
# =====================================================

vif(
  spillover_model
)

