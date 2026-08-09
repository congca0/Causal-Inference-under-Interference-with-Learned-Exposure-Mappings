 

all_sigma_data <- list()

 

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
  
 
  
  gaussian_kernel <- exp(
    
    -(distance_matrix^2) /
      
      (2 * sig^2)
  )
  
  diag(gaussian_kernel) <- 0
  
 
  
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
    
    spillover_results[[as.character(d)]] <- day_data
    
  }   # END DATE LOOP
  
 
  
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
  
 
  
  all_sigma_data[[as.character(sig)]] <- sigma_data
  
 
  
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
    
    RMSE = rmse(
      sigma_data$pm25,
      pred
    ),
    
    R2 = summary(fit)$r.squared,
    
    spillover_beta =
      coef(fit)["spillover"]
  )
  
}   # END SIGMA LOOP

 

sigma_results <- bind_rows(
  sigma_results
)

print(sigma_results)

 

best_sigma <- sigma_results %>%
  
  arrange(AIC) %>%
  
  slice(1)

print(best_sigma)
 

best_sigma_value <- best_sigma$sigma

print(best_sigma_value)

 

model_data <- all_sigma_data[[as.character(best_sigma_value)]]

 

cat(
  "model_data rows:",
  nrow(model_data),
  "\n"
)

head(model_data)

 

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

 

baseline_model <- lm(
  
  pm25 ~
    
    pm25_lag1 +
    
    pm25_lag2 +
    
    wind_speed +
    
    temperature,
  
  data = model_data
)

summary(baseline_model)

 

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

 

AIC(
  baseline_model,
  spillover_model
)

 

vif(
  spillover_model
)

