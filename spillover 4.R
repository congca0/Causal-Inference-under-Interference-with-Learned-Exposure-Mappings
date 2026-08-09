 

library(dplyr)
library(ggplot2)
library(tidyr)
library(broom)
library(spdep)
library(lmtest)
library(sandwich)
library(modelsummary)
library(patchwork)
library(scales)
library(grid)

 
 
model_data <- model_data %>%
  
  mutate(
    
    spillover =
      as.numeric(spillover),
    
    pm25 =
      as.numeric(pm25),
    
    pm25_lag1 =
      as.numeric(pm25_lag1),
    
    pm25_lag2 =
      as.numeric(pm25_lag2),
    
    wind_speed =
      as.numeric(wind_speed),
    
    temperature =
      as.numeric(temperature)
  )

 

model_data <- model_data %>%
  
  drop_na(
    
    pm25,
    
    pm25_lag1,
    
    spillover,
    
    wind_speed,
    
    temperature
  )

 

baseline_model <- lm(
  
  pm25 ~
    
    pm25_lag1 +
    
    pm25_lag2 +
    
    wind_speed +
    
    temperature,
  
  data = model_data
)

summary(
  baseline_model
)

 

spillover_model <- lm(
  
  pm25 ~
    
    pm25_lag1 +
    
    pm25_lag2 +
    
    spillover +
    
    wind_speed +
    
    temperature,
  
  data = model_data
)

summary(
  spillover_model
)

 

spillover_fe_model <- lm(
  
  pm25 ~
    
    pm25_lag1 +
    
    spillover +
    
    wind_speed +
    
    temperature +
    
    factor(county_name),
  
  data = model_data
)

summary(
  spillover_fe_model
)

 

two_way_fe_model <- lm(
  
  pm25 ~
    
    pm25_lag1 +
    
    spillover +
    
    wind_speed +
    
    temperature +
    
    factor(county_name) +
    
    factor(date_local),
  
  data = model_data
)

summary(
  two_way_fe_model
)

 
 

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

 

model_data$predicted <- predict(
  
  two_way_fe_model
)

model_data$residuals <- residuals(
  
  two_way_fe_model
)

 

moran_results <- list()

for(d in sort(unique(model_data$date_local))){
  
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
  
  valid <- !is.na(idx)
  
  day_df <- day_df[
    valid,
  ]
  
  idx <- idx[
    valid
  ]
  
  coords_sub <- coords_matrix[
    
    idx,
    ,
    
    drop = FALSE
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
    
    date_local =
      as.Date(d),
    
    moran_I =
      unname(
        moran_out$estimate[1]
      ),
    
    expectation =
      unname(
        moran_out$estimate[2]
      ),
    
    variance =
      unname(
        moran_out$estimate[3]
      ),
    
    p_value =
      moran_out$p.value
  )
}

moran_results <- bind_rows(
  
  moran_results
)

print(
  moran_results
)

 

fraction_sig <- mean(
  
  moran_results$p_value < 0.05,
  
  na.rm = TRUE
)

print(
  fraction_sig
)

 

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
    
    .groups = "drop"
  )

print(
  county_residuals
)

 

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
 
theme_set(
  
  theme_classic(
    
    base_size = 14
  )
)

 

p1 <- ggplot(
  
  model_data,
  
  aes(
    
    x = predicted,
    
    y = pm25
  )
) +
  
  geom_point(
    
    alpha = 0.22,
    
    size = 1.2
  ) +
  
  geom_smooth(
    
    method = "lm",
    
    se = FALSE,
    
    linewidth = 0.8
  ) +
  
  labs(
    
    x = "Predicted PM2.5",
    
    y = "Observed PM2.5"
  )

 

p2 <- ggplot(
  
  model_data,
  
  aes(
    
    x = predicted,
    
    y = residuals
  )
) +
  
  geom_point(
    
    alpha = 0.22,
    
    size = 1.2
  ) +
  
  geom_hline(
    
    yintercept = 0,
    
    linetype = "dashed",
    
    linewidth = 0.7
  ) +
  
  labs(
    
    x = "Predicted PM2.5",
    
    y = "Residuals"
  )

 

p3 <- ggplot(
  
  model_data,
  
  aes(
    x = residuals
  )
) +
  
  geom_histogram(
    
    bins = 30,
    
    linewidth = 0.2
  ) +
  
  labs(
    
    x = "Residuals",
    
    y = "Count"
  )

 

p4 <- ggplot(
  
  model_data,
  
  aes(
    
    x = spillover,
    
    y = pm25
  )
) +
  
  geom_point(
    
    alpha = 0.22,
    
    size = 1.2
  ) +
  
  geom_smooth(
    
    method = "lm",
    
    se = FALSE,
    
    linewidth = 0.8
  ) +
  
  labs(
    
    x = "Gaussian Spillover",
    
    y = "Observed PM2.5"
  )

 

p5 <- ggplot(
  
  moran_results,
  
  aes(
    
    x = date_local,
    
    y = moran_I
  )
) +
  
  geom_line(
    linewidth = 0.8
  ) +
  
  geom_point(
    size = 1.5
  ) +
  
  geom_hline(
    
    yintercept = 0,
    
    linetype = "dashed",
    
    linewidth = 0.7
  ) +
  
  scale_x_date(
    
    date_breaks = "7 days",
    
    labels = date_format("%b %d")
  ) +
  
  labs(
    
    x = "Date",
    
    y = "Moran's I"
  ) +
  
  theme(
    
    axis.text.x =
      element_text(
        
        angle = 45,
        
        hjust = 1
      )
  )

 

print(
  sigma_results
)

p6 <- ggplot(
  
  sigma_results,
  
  aes(
    
    x = sigma,
    
    y = R2
  )
) +
  
  geom_line(
    linewidth = 0.8
  ) +
  
  geom_point(
    size = 2
  ) +
  
  labs(
    
    x = expression(sigma),
    
    y = expression(R^2)
  )

 

library(patchwork)

final_figure <- (
  
  (p1 | p2 | p3) /
    (p4 | p5 | p6)
  
) +
  
  plot_annotation(
    
    tag_levels = "A"
  )

 

final_figure <- final_figure +
  
  plot_layout(
    
    guides = "collect"
  )

 

print(
  final_figure
)

 

ggsave(
  
  filename =
    "nature_style_transport_figure.png",
  
  plot =
    final_figure,
  
  width = 16,
  
  height = 9,
  
  dpi = 600,
  
  bg = "white"
)

 
 

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
  
  coef_omit =
    "factor\\(",
  
  output =
    "modelsummary.html"
)

 

 

cat(
  
  "FINAL PAPER RESULTS\n"
)

 

print(
  summary_table
)

cat(
  
  "\nSpillover coefficient:\n"
)

print(
  coef(two_way_fe_model)["spillover"]
)

cat(
  
  "\nFraction significant Moran tests:\n"
)

print(
  fraction_sig
)

cat(
  
  "\nAll outputs successfully exported.\n"
)
