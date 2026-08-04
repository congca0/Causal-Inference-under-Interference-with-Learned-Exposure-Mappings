spillover_fe_model <- lm(
  
  pm25 ~
    
    pm25_lag1 +
    
    spillover +
    
    wind_speed +
    
    temperature +
    
    factor(county_name),
  
  data = model_data
)

summary(spillover_fe_model)


par(mfrow = c(2,2))

plot(spillover_model)




write.csv(
  merged_data,
  "merged_data.csv"
)





library(dplyr)

library(janitor)

county_coords <- pm25 %>%
  
  clean_names() %>%
  
  filter(
    state_name == "California"
  ) %>%
  
  group_by(county_name) %>%
  
  summarise(
    
    latitude = mean(latitude),
    
    longitude = mean(longitude),
    
    .groups = "drop"
  )

head(county_coords)

write.csv(
  
  county_coords,
  
  "county_coordinates.csv",
  
  row.names = FALSE
)

write.csv(

  merged_data,

  "merged_data.csv",

  row.names = FALSE
)

