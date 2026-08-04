# =====================================================
# GAUSSIAN SPILLOVER BENCHMARK
# FULL RESULTS PIPELINE
# =====================================================

library(dplyr)

library(tidyr)

library(janitor)

library(geosphere)

library(ggplot2)

library(readr)

library(car)

library(broom)

# =====================================================
# RESULTS STORAGE
# =====================================================

results_table <- data.frame()

coef_table <- data.frame()

model_list <- list()

# =====================================================
# STEP 0
# CLEAN DATA
# =====================================================

merged_data <- merged_data %>%
  
  clean_names()

# -----------------------------------------------------
# CHECK STRUCTURE
# -----------------------------------------------------

str(merged_data)

# -----------------------------------------------------
# NUMERIC CONVERSION
# -----------------------------------------------------

numeric_vars <- c(
  
  "pm25",
  "wind_speed",
  "temperature",
  "pm25_lag1",
  "pm25_lag2"
)

numeric_vars <- numeric_vars[
  
  numeric_vars %in%
    
    names(merged_data)
]

merged_data <- merged_data %>%
  
  mutate(
    
    across(
      
      all_of(numeric_vars),
      
      ~ readr::parse_number(
        as.character(.)
      )
    )
  )

# -----------------------------------------------------
# CHECK MISSINGNESS
# -----------------------------------------------------

cat(
  
  "\n=============================\n"
)

cat(
  
  "MISSINGNESS\n"
)

cat(
  
  "=============================\n"
)

print(
  colSums(
    is.na(merged_data)
  )
)

# =====================================================
# STEP 1
# CREATE TEMPORAL LAGS
# =====================================================

merged_data <- merged_data %>%
  
  arrange(
    county_name,
    date_local
  ) %>%
  
  group_by(county_name) %>%
  
  mutate(
    
    pm25_lag1 = lag(pm25, 1),
    
    pm25_lag2 = lag(pm25, 2)
  ) %>%
  
  ungroup()

# =====================================================
# STEP 2
# COUNTY COORDINATES
# =====================================================

county_coords <- county_coords %>%
  
  distinct(
    county_name,
    .keep_all = TRUE
  )

# =====================================================
# STEP 3
# COMMON COUNTIES
# =====================================================

common_counties <- intersect(
  
  unique(merged_data$county_name),
  
  unique(county_coords$county_name)
)

merged_data <- merged_data %>%
  
  filter(
    county_name %in%
      common_counties
  )

county_coords <- county_coords %>%
  
  filter(
    county_name %in%
      common_counties
  )

# =====================================================
# STEP 4
# FIXED COUNTY ORDER
# =====================================================

county_coords <- county_coords %>%
  
  arrange(county_name)

county_levels <- sort(
  
  unique(
    county_coords$county_name
  )
)

# =====================================================
# STEP 5
# DISTANCE MATRIX
# =====================================================

coords_matrix <- county_coords %>%
  
  select(
    longitude,
    latitude
  ) %>%
  
  as.matrix()

distance_matrix <- distm(
  
  coords_matrix,
  
  fun = distHaversine
)

distance_matrix <- distance_matrix / 1000

rownames(distance_matrix) <- county_coords$county_name

colnames(distance_matrix) <- county_coords$county_name

# =====================================================
# STEP 6
# GAUSSIAN KERNEL
# =====================================================

sigma <- 100

gaussian_kernel <- exp(
  
  -(distance_matrix^2) /
    
    (2 * sigma^2)
)

diag(gaussian_kernel) <- 0

# =====================================================
# STEP 7
# PANEL DATA
# =====================================================

panel_data <- merged_data %>%
  
  arrange(
    date_local,
    county_name
  )

unique_dates <- sort(
  
  unique(panel_data$date_local)
)

spillover_results <- list()

# =====================================================
# STEP 8
# SPILLOVER LOOP
# =====================================================

for(d in unique_dates){
  
  cat(
    
    "Processing:",
    
    as.character(d),
    
    "\n"
  )
  
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
  
  stopifnot(
    
    all(
      as.character(day_data$county_name) ==
        county_levels
    )
  )
  
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
  
  day_data$gaussian_spillover <- as.numeric(
    spillover
  )
  
  spillover_results[[as.character(d)]] <- day_data
}

# =====================================================
# STEP 9
# FINAL DATA
# =====================================================

spillover_data <- bind_rows(
  spillover_results
)

cat(
  
  "\nspillover_data rows:",
  
  nrow(spillover_data),
  
  "\n"
)

# =====================================================
# STEP 10
# MODEL DATA
# =====================================================

model_data <- spillover_data %>%
  
  filter(
    
    !is.na(pm25),
    
    !is.na(pm25_lag1),
    
    !is.na(pm25_lag2),
    
    !is.na(wind_speed),
    
    !is.na(temperature),
    
    !is.na(gaussian_spillover)
  )

cat(
  
  "model_data rows:",
  
  nrow(model_data),
  
  "\n"
)

# =====================================================
# STEP 11
# CORRELATION MATRIX
# =====================================================

cor_matrix <- model_data %>%
  
  select(
    
    pm25,
    
    pm25_lag1,
    
    pm25_lag2,
    
    gaussian_spillover,
    
    wind_speed,
    
    temperature
  ) %>%
  
  cor(
    use = "complete.obs"
  )

cat(
  
  "\n=============================\n"
)

cat(
  
  "CORRELATION MATRIX\n"
)

cat(
  
  "=============================\n"
)

print(cor_matrix)

write.csv(
  
  cor_matrix,
  
  "correlation_matrix.csv"
)

# =====================================================
# STEP 12
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

model_list[["baseline_model"]] <- baseline_model

# -----------------------------------------------------
# SAVE METRICS
# -----------------------------------------------------

baseline_metrics <- data.frame(
  
  model = "baseline_model",
  
  r_squared =
    summary(baseline_model)$r.squared,
  
  adj_r_squared =
    summary(baseline_model)$adj.r.squared,
  
  aic =
    AIC(baseline_model),
  
  bic =
    BIC(baseline_model),
  
  rmse =
    sqrt(
      mean(
        residuals(baseline_model)^2
      )
    )
)

results_table <- bind_rows(
  
  results_table,
  
  baseline_metrics
)

# -----------------------------------------------------
# SAVE COEFFICIENTS
# -----------------------------------------------------

baseline_coef <- broom::tidy(
  
  baseline_model
) %>%
  
  mutate(
    model = "baseline_model"
  )

coef_table <- bind_rows(
  
  coef_table,
  
  baseline_coef
)

# =====================================================
# STEP 13
# SPILLOVER MODEL
# =====================================================

spillover_model <- lm(
  
  pm25 ~
    
    pm25_lag1 +
    
    pm25_lag2 +
    
    gaussian_spillover +
    
    wind_speed +
    
    temperature,
  
  data = model_data
)

summary(spillover_model)

model_list[["spillover_model"]] <- spillover_model

# -----------------------------------------------------
# SAVE METRICS
# -----------------------------------------------------

spillover_metrics <- data.frame(
  
  model = "spillover_model",
  
  r_squared =
    summary(spillover_model)$r.squared,
  
  adj_r_squared =
    summary(spillover_model)$adj.r.squared,
  
  aic =
    AIC(spillover_model),
  
  bic =
    BIC(spillover_model),
  
  rmse =
    sqrt(
      mean(
        residuals(spillover_model)^2
      )
    )
)

results_table <- bind_rows(
  
  results_table,
  
  spillover_metrics
)

# -----------------------------------------------------
# SAVE COEFFICIENTS
# -----------------------------------------------------

spillover_coef <- broom::tidy(
  
  spillover_model
) %>%
  
  mutate(
    model = "spillover_model"
  )

coef_table <- bind_rows(
  
  coef_table,
  
  spillover_coef
)

# =====================================================
# STEP 14
# AIC COMPARISON
# =====================================================

aic_table <- AIC(
  
  baseline_model,
  
  spillover_model
)

print(aic_table)

write.csv(
  
  aic_table,
  
  "aic_comparison.csv",
  
  row.names = FALSE
)

# =====================================================
# STEP 15
# VIF
# =====================================================

vif_results <- vif(
  spillover_model
)

cat(
  
  "\n=============================\n"
)

cat(
  
  "VIF RESULTS\n"
)

cat(
  
  "=============================\n"
)

print(vif_results)

write.csv(
  
  as.data.frame(vif_results),
  
  "vif_results.csv"
)

# =====================================================
# STEP 16
# PREDICTIONS
# =====================================================

model_data$predicted_pm25 <- predict(
  
  spillover_model
)

model_data$residuals <- residuals(
  
  spillover_model
)

# =====================================================
# STEP 17
# OBSERVED VS PREDICTED
# =====================================================

fig_obs_pred <- ggplot(
  
  model_data,
  
  aes(
    
    x = pm25,
    
    y = predicted_pm25
  )
) +
  
  geom_point(alpha=0.5) +
  
  geom_abline(
    
    slope=1,
    
    intercept=0,
    
    color="red"
  ) +
  
  theme_minimal() +
  
  ggtitle(
    "Observed vs Predicted PM2.5"
  )

print(fig_obs_pred)

# =====================================================
# STEP 18
# RESIDUAL HISTOGRAM
# =====================================================

fig_resid <- ggplot(
  
  model_data,
  
  aes(
    x = residuals
  )
) +
  
  geom_histogram(
    
    bins=40,
    
    fill="skyblue"
  ) +
  
  theme_minimal() +
  
  ggtitle(
    "Residual Distribution"
  )

print(fig_resid)

# =====================================================
# STEP 19
# SAVE FIGURES
# =====================================================

ggsave(
  
  "observed_vs_predicted.png",
  
  fig_obs_pred,
  
  width=8,
  
  height=6
)

ggsave(
  
  "residual_distribution.png",
  
  fig_resid,
  
  width=8,
  
  height=6
)

# =====================================================
# STEP 20
# SAVE TABLES
# =====================================================

write.csv(
  
  results_table,
  
  "results_table.csv",
  
  row.names = FALSE
)

write.csv(
  
  coef_table,
  
  "coefficient_table.csv",
  
  row.names = FALSE
)

write.csv(
  
  model_data,
  
  "final_model_data.csv",
  
  row.names = FALSE
)

# =====================================================
# STEP 21
# FINAL PRINT
# =====================================================

cat(
  
  "\n=====================================\n"
)

cat(
  
  "FINAL MODEL RESULTS\n"
)

cat(
  
  "=====================================\n"
)

print(results_table)

cat(
  
  "\n=====================================\n"
)

cat(
  
  "COEFFICIENTS\n"
)

cat(
  
  "=====================================\n"
)

print(coef_table)

cat(
  
  "\n=====================================\n"
)

cat(
  
  "FINAL SAMPLE SIZES\n"
)

cat(
  
  "=====================================\n"
)

cat(
  
  "merged_data:",
  
  nrow(merged_data),
  
  "\n"
)

cat(
  
  "panel_data:",
  
  nrow(panel_data),
  
  "\n"
)

cat(
  
  "spillover_data:",
  
  nrow(spillover_data),
  
  "\n"
)

cat(
  
  "model_data:",
  
  nrow(model_data),
  
  "\n"
)

cat(
  
  "\nALL RESULTS SAVED\n"
)

# =====================================================
# END
# =====================================================