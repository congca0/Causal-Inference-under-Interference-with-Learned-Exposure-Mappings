# =====================================================
# FINAL PAPER 1 PIPELINE
# TWO-WAY FIXED EFFECTS + DIAGNOSTICS
# =====================================================

library(dplyr)

library(ggplot2)

library(spdep)

library(broom)

library(lmtest)

library(sandwich)

library(modelsummary)

library(patchwork)

# =====================================================
# STEP 1
# TWO-WAY FIXED EFFECTS MODEL
# =====================================================
model_data$gaussian_spillover <- as.numeric(
       
      model_data$gaussian_spillover
  )
two_way_fe_model <- lm(
  
  pm25 ~
    
    pm25_lag1 +
    
    gaussian_spillover +
    
    wind_speed +
    
    temperature +
    
    factor(county_name) +
    
    factor(date_local),
  
  data = model_data
)

summary(
  two_way_fe_model
)

# =====================================================
# STEP 2
# CLUSTER-ROBUST STANDARD ERRORS
# =====================================================

robust_results <- coeftest(
  
  two_way_fe_model,
  
  vcov =
    vcovCL(
      
      two_way_fe_model,
      
      cluster =
        ~ county_name
    )
)

print(
  robust_results
)

# =====================================================
# STEP 3
# SAVE PREDICTIONS + RESIDUALS
# =====================================================

model_data$predicted <- predict(
  
  two_way_fe_model
)

model_data$residuals <- residuals(
  
  two_way_fe_model
)

# =====================================================
# STEP 4
# BASIC DIAGNOSTICS
# =====================================================

par(
  mfrow = c(2,2)
)

plot(
  two_way_fe_model
)

# =====================================================
# STEP 5
# RESIDUAL HISTOGRAM
# =====================================================

fig_resid_hist <- ggplot(
  
  model_data,
  
  aes(
    x = residuals
  )
) +
  
  geom_histogram(
    
    bins = 40
  ) +
  
  theme_bw() +
  
  labs(
    
    title =
      "Residual Distribution",
    
    x = "Residual",
    
    y = "Count"
  )

print(
  fig_resid_hist
)

# =====================================================
# STEP 6
# OBSERVED VS PREDICTED
# =====================================================

fig_obs_pred <- ggplot(
  
  model_data,
  
  aes(
    
    x = predicted,
    
    y = pm25
  )
) +
  
  geom_point(
    
    alpha = 0.5
  ) +
  
  geom_smooth(
    
    method = "lm",
    
    color = "red"
  ) +
  
  theme_bw() +
  
  labs(
    
    title =
      "Observed vs Predicted",
    
    x = "Predicted PM2.5",
    
    y = "Observed PM2.5"
  )

print(
  fig_obs_pred
)

# =====================================================
# STEP 7
# RESIDUALS VS FITTED
# =====================================================

fig_resid_fit <- ggplot(
  
  model_data,
  
  aes(
    
    x = predicted,
    
    y = residuals
  )
) +
  
  geom_point(
    
    alpha = 0.5
  ) +
  
  geom_hline(
    
    yintercept = 0,
    
    color = "red",
    
    linetype = "dashed"
  ) +
  
  theme_bw() +
  
  labs(
    
    title =
      "Residuals vs Fitted",
    
    x = "Predicted",
    
    y = "Residuals"
  )

print(
  fig_resid_fit
)

# =====================================================
# STEP 8
# COUNTY-LEVEL RMSE
# =====================================================

county_residuals <- model_data %>%
  
  group_by(
    county_name
  ) %>%
  
  summarise(
    
    rmse =
      sqrt(
        mean(
          residuals^2
        )
      ),
    
    mean_abs_residual =
      mean(
        abs(residuals)
      ),
    
    .groups = "drop"
  )

print(
  county_residuals
)

# =====================================================
# STEP 9
# COUNTY RMSE FIGURE
# =====================================================

fig_county_rmse <- ggplot(
  
  county_residuals,
  
  aes(
    
    x =
      reorder(
        county_name,
        rmse
      ),
    
    y = rmse
  )
) +
  
  geom_col() +
  
  coord_flip() +
  
  theme_bw() +
  
  labs(
    
    title =
      "County-Level RMSE",
    
    x = "County",
    
    y = "RMSE"
  )

print(
  fig_county_rmse
)

# =====================================================
# STEP 10
# MORAN'S I
# =====================================================

moran_results <- list()

for(d in unique(model_data$date_local)){
  
  cat(
    
    "Moran:",
    
    as.character(d),
    
    "\n"
  )
  
  day_df <- model_data %>%
    
    filter(
      date_local == d
    ) %>%
    
    arrange(
      county_name
    )
  
  if(nrow(day_df) < 5){
    
    next
  }
  
  idx <- match(
    
    day_df$county_name,
    
    county_levels
  )
  
  coords_sub <- coords_matrix[
    idx,
  ]
  
  neighbors_sub <- knn2nb(
    
    knearneigh(
      
      coords_sub,
      
      k = 4
    )
  )
  
  weights_sub <- nb2listw(
    
    neighbors_sub,
    
    style = "W",
    
    zero.policy = TRUE
  )
  
  moran_out <- moran.test(
    
    day_df$residuals,
    
    weights_sub,
    
    zero.policy = TRUE
  )
  
  
  moran_results[[as.character(d)]] <- tibble(
    
    date_local = d,
    
    moran_I =
      moran_out$estimate[1],
    
    expectation =
      moran_out$estimate[2],
    
    variance =
      moran_out$estimate[3],
    
    p_value =
      moran_out$p.value
  )

# =====================================================
# STEP 11
# FINAL MORAN RESULTS
# =====================================================

moran_results <- bind_rows(
  moran_results
)

print(
  moran_results
)

# =====================================================
# STEP 12
# FRACTION SIGNIFICANT
# =====================================================

fraction_sig <- mean(
  
  moran_results$p_value < 0.05,
  
  na.rm = TRUE
)

print(
  fraction_sig
)

# =====================================================
# STEP 13
# MORAN FIGURE
# =====================================================

fig_moran <- ggplot(
  
  moran_results,
  
  aes(
    
    x = date_local,
    
    y = moran_I
  )
) +
  
  geom_line(
    linewidth = 1
  ) +
  
  geom_point(
    size = 2
  ) +
  
  geom_hline(
    
    yintercept = 0,
    
    color = "red",
    
    linetype = "dashed"
  ) +
  
  theme_bw() +
  
  labs(
    
    title =
      "Residual Moran's I",
    
    x = "Date",
    
    y = "Moran's I"
  )

print(
  fig_moran
)

# =====================================================
# STEP 14
# MODEL SUMMARY TABLE
# =====================================================

summary_table <- tibble(
  
  model = c(
    
    "Baseline",
    
    "Spillover",
    
    "County FE",
    
    "Two-Way FE"
  ),
  
  R2 = c(
    
    summary(baseline_model)$r.squared,
    
    summary(spillover_model)$r.squared,
    
    summary(spillover_fe_model)$r.squared,
    
    summary(two_way_fe_model)$r.squared
  ),
  
  AIC = c(
    
    AIC(baseline_model),
    
    AIC(spillover_model),
    
    AIC(spillover_fe_model),
    
    AIC(two_way_fe_model)
  )
)

print(
  summary_table
)

# =====================================================
# STEP 15
# MODEL SUMMARY EXPORT
# =====================================================

modelsummary(
  
  list(
    
    "Baseline" =
      baseline_model,
    
    "Spillover" =
      spillover_model,
    
    "County FE" =
      spillover_fe_model,
    
    "Two-Way FE" =
      two_way_fe_model
  ),
  
  output =
    "modelsummary.html"
)

# =====================================================
# STEP 16
# COEFFICIENT PLOT
# =====================================================

coef_df <- tidy(
  
  two_way_fe_model
)

coef_df <- coef_df %>%
  
  filter(
    
    term %in% c(
      
      "pm25_lag1",
      
      "spillover",
      
      "wind_speed",
      
      "temperature"
    )
  )

fig_coef <- ggplot(
  
  coef_df,
  
  aes(
    
    x = term,
    
    y = estimate
  )
) +
  
  geom_point(
    size = 3
  ) +
  
  geom_errorbar(
    
    aes(
      
      ymin =
        estimate -
        1.96*std.error,
      
      ymax =
        estimate +
        1.96*std.error
    ),
    
    width = 0.1
  ) +
  
  theme_bw() +
  
  labs(
    
    title =
      "Two-Way FE Coefficients",
    
    x = "",
    
    y = "Estimate"
  )

print(
  fig_coef
)

# =====================================================
# STEP 17
# SAVE FIGURES
# =====================================================

ggsave(
  
  "figure_obs_pred.png",
  
  fig_obs_pred,
  
  width = 8,
  
  height = 6
)

ggsave(
  
  "figure_residual_hist.png",
  
  fig_resid_hist,
  
  width = 8,
  
  height = 6
)

ggsave(
  
  "figure_residual_fit.png",
  
  fig_resid_fit,
  
  width = 8,
  
  height = 6
)

ggsave(
  
  "figure_county_rmse.png",
  
  fig_county_rmse,
  
  width = 8,
  
  height = 8
)

ggsave(
  
  "figure_moran.png",
  
  fig_moran,
  
  width = 10,
  
  height = 5
)

ggsave(
  
  "figure_coef.png",
  
  fig_coef,
  
  width = 8,
  
  height = 6
)

# =====================================================
# STEP 18
# EXPORT TABLES
# =====================================================

write.csv(
  
  summary_table,
  
  "summary_table.csv",
  
  row.names = FALSE
)

write.csv(
  
  moran_results,
  
  "moran_results.csv",
  
  row.names = FALSE
)

write.csv(
  
  county_residuals,
  
  "county_residuals.csv",
  
  row.names = FALSE
)

# =====================================================
# STEP 19
# EXPORT FINAL DATA
# =====================================================

export_data <- model_data %>%
  
  select(
    
    county_name,
    
    date_local,
    
    pm25,
    
    pm25_lag1,
    
    spillover,
    
    wind_speed,
    
    temperature,
    
    predicted,
    
    residuals
  )

write.csv(
  
  export_data,
  
  "epa_pm25_transport.csv",
  
  row.names = FALSE
)

# =====================================================
# STEP 20
# FINAL SUMMARY
# =====================================================

cat(
  
  "\n=============================\n"
)

cat(
  
  "FINAL PAPER 1 RESULTS\n"
)

cat(
  
  "=============================\n"
)

cat(
  
  "Two-Way FE R2:",
  
  summary(two_way_fe_model)$r.squared,
  
  "\n"
)

cat(
  
  "Two-Way FE AIC:",
  
  AIC(two_way_fe_model),
  
  "\n"
)

cat(
  
  "Spillover Coefficient:",
  
  coef(two_way_fe_model)["spillover"],
  
  "\n"
)

cat(
  
  "Fraction Significant Moran Tests:",
  
  fraction_sig,
  
  "\n"
)

cat(
  
  "All outputs exported.\n"
)