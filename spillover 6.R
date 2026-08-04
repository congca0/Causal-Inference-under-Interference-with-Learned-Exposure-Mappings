# =====================================================
# FIXED VERSION
#
# YOU HAD A SYNTAX ERROR:
#
# WRONG:
# spillover_results[
#   [as.character(d)]
#
# CORRECT:
# spillover_results[[as.character(d)]] <- day_data
#
# =====================================================

# =====================================================
# STEP 1
# SIGMA SENSITIVITY ANALYSIS
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
  # BUILD SPILLOVER
  # --------------------------------------------------
  
  spillover_results <- list()
  
  for(d in unique_dates){
    
    day_data <- panel_data %>%
      
      filter(
        date_local == d
      )
    
    day_data <- tibble(
      
      county_name = county_levels
      
    ) %>%
      
      left_join(
        
        day_data,
        
        by = "county_name"
      )
    
    day_data$county_name <- factor(
      
      day_data$county_name,
      
      levels = county_levels
    )
    
    day_data <- day_data %>%
      
      arrange(county_name)
    
    pm_vector <- day_data$pm25
    
    valid_idx <- !is.na(pm_vector)
    
    sub_kernel <- gaussian_kernel[
      ,
      valid_idx
    ]
    
    sub_pm <- pm_vector[
      valid_idx
    ]
    
    row_weights <- rowSums(
      sub_kernel
    )
    
    row_weights[
      row_weights == 0
    ] <- NA
    
    spillover <- (
      
      sub_kernel %*%
        sub_pm
      
    ) / row_weights
    
    day_data$spillover <- as.numeric(
      spillover
    )
    
    # =================================================
    # CORRECT LIST ASSIGNMENT
    # =================================================
    
    spillover_results[[as.character(d)]] <- day_data
    
  }   # END DATE LOOP
  
  # --------------------------------------------------
  # COMBINE
  # --------------------------------------------------
  
  sigma_data <- bind_rows(
    spillover_results
  )
  
  sigma_data <- sigma_data %>%
    
    filter(
      
      !is.na(pm25),
      
      !is.na(pm25_lag1),
      
      !is.na(pm25_lag2),
      
      !is.na(wind_speed),
      
      !is.na(temperature),
      
      !is.na(spillover)
    )
  
  # --------------------------------------------------
  # MODEL
  # --------------------------------------------------
  
  fit <- lm(
    
    pm25 ~
      
      pm25_lag1 +
      
      pm25_lag2 +
      
      spillover +
      
      wind_speed +
      
      temperature,
    
    data = sigma_data
  )
  
  pred <- predict(fit)
  
  sigma_results[[as.character(sig)]] <- tibble(
    
    sigma = sig,
    
    AIC = AIC(fit),
    
    RMSE = Metrics::rmse(
      sigma_data$pm25,
      pred
    ),
    
    R2 = summary(fit)$r.squared,
    
    spillover_beta =
      coef(fit)["spillover"]
  )
  
}   # END SIGMA LOOP

# =====================================================
# FINAL RESULTS
# =====================================================

sigma_results <- bind_rows(
  sigma_results
)

print(sigma_results)

# =====================================================
# VISUALIZE
# =====================================================
ggplot(
  sigma_results,
  
  aes(
    x = sigma,
    y = RMSE
  )
) +
  
  geom_line(linewidth = 1) +
  
  geom_point(size = 3) +
  
  theme_bw()




ggplot(
  sigma_results,
  
  aes(
    x = sigma,
    y = R2
  )
) +
  
  geom_line(linewidth = 1) +
  
  geom_point(size = 3) +
  
  theme_bw()


