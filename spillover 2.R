 

library(dplyr)

library(tidyr)

library(janitor)

library(ggplot2)

library(readr)

library(geosphere)

library(car)

library(spdep)
 

merged_data <- merged_data %>%
  
  clean_names()

 

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
      
      ~ parse_number(
        as.character(.)
      )
    )
  )
 

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

 

cat(
  "merged_data rows:",
  nrow(merged_data),
  "\n"
)

panel_check <- merged_data %>%
  
  count(
    county_name,
    date_local
  ) %>%
  
  filter(
    n > 1
  )

print(panel_check)

 

county_coords <- county_coords %>%
  
  clean_names()

county_coords <- county_coords %>%
  
  distinct(
    county_name,
    .keep_all = TRUE
  )

 

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

 

county_coords <- county_coords %>%
  
  arrange(
    county_name
  )

county_levels <- county_coords$county_name

 

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

# meters -> km
distance_matrix <- distance_matrix / 1000

rownames(distance_matrix) <- county_levels

colnames(distance_matrix) <- county_levels

 

image(distance_matrix)

title("County Distance Matrix")

 

sigma <- 100

gaussian_kernel <- exp(
  
  -(distance_matrix^2) /
    
    (2 * sigma^2)
)

diag(gaussian_kernel) <- 0

 

panel_data <- merged_data %>%
  
  arrange(
    date_local,
    county_name
  )

 

unique_dates <- sort(
  unique(panel_data$date_local)
)

spillover_results <- list()

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
    
    arrange(
      county_name
    )
  
 
  
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

 

spillover_data <- bind_rows(
  spillover_results
)

cat(
  "spillover_data rows:",
  nrow(spillover_data),
  "\n"
)

 

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

 

colSums(
  is.na(model_data)
)

 
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

print(cor_matrix)

 
ggplot(
  model_data,
  
  aes(
    x = gaussian_spillover,
    y = pm25
  )
) +
  
  geom_point(alpha = 0.5) +
  
  geom_smooth(
    method = "lm"
  ) +
  
  theme_bw()
 

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
    
    gaussian_spillover +
    
    wind_speed +
    
    temperature,
  
  data = model_data
)

summary(spillover_model)

 

AIC(
  baseline_model,
  spillover_model
)

 

vif(spillover_model)

 

model_data$residuals <- residuals(
  spillover_model
)

 

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
    
   )
  
 
  moran_out <- moran.test(
    
    day_df$residuals,
    
    weights_sub,
    
    zero.policy = TRUE
  )
  
 
 
  
  moran_results[[as.character(d)]] <- tibble(
    
    date_local = d,
    
    n_counties = nrow(day_df),
    
    moran_I =
      moran_out$estimate[1],
    
    expectation =
      moran_out$estimate[2],
    
    variance =
      moran_out$estimate[3],
    
    p_value =
      moran_out$p.value
  )
}

 

moran_results <- bind_rows(
  moran_results
)

print(moran_results)

 

summary(
  moran_results$moran_I
)

summary(
  moran_results$p_value
)

 
mean(
  moran_results$p_value < 0.05,
  na.rm = TRUE
)
 

ggplot(
  moran_results,
  
  aes(
    x = date_local,
    y = moran_I
  )
) +
  
  geom_line() +
  
  geom_point() +
  
  theme_bw()
 
 
