library(data.table)

library(dplyr)

library(janitor)

# =====================================================
# READ
# =====================================================

pm25 <- fread("daily_88101_2024.csv")

# =====================================================
# CLEAN COLUMN NAMES
# =====================================================

pm25 <- clean_names(pm25)

# =====================================================
# CHECK NAMES
# =====================================================

names(pm25)

# =====================================================
# FILTER CALIFORNIA
# =====================================================

pm25_ca <- pm25 %>%
  
  filter(state_name == "California")

# =====================================================
# DATE
# =====================================================

pm25_ca$date_local <- as.Date(
  pm25_ca$date_local
)

# =====================================================
# JANUARY ONLY
# =====================================================

pm25_ca_jan <- pm25_ca %>%
  
  filter(
    
    date_local >= as.Date("2024-01-01") &
      
      date_local <= as.Date("2024-01-31")
  )

# =====================================================
# SELECT VARIABLES
# =====================================================

pm25_ca_jan <- pm25_ca_jan %>%
  
  select(
    
    date_local,
    
    state_name,
    
    county_name,
    
    city_name,
    
    latitude,
    
    longitude,
    
    arithmetic_mean,
    
    aqi
  )

# =====================================================
# CHECK
# =====================================================

head(pm25_ca_jan)

summary(pm25_ca_jan)
# =====================================================
# CHECK DATA
# =====================================================

head(pm25_ca_jan)

summary(pm25_ca_jan)

# =====================================================
# NUMBER OF STATIONS
# =====================================================

pm25_ca_jan %>%
  
  distinct(
    latitude,
    longitude
  ) %>%
  
  nrow()

# =====================================================
# BASIC MAP
# =====================================================
library(ggplot2)

ggplot(
  
  pm25_ca_jan,
  
  aes(
    
    x = longitude,
    
    y = latitude,
    
    color = arithmetic_mean
  )
  
) +
  
  geom_point(alpha = 0.7) +
  
  scale_color_viridis_c() +
  
  theme_minimal() +
  
  ggtitle("California PM2.5 January 2024")

# =====================================================
# SAVE
# =====================================================

write.csv(
  
  pm25_ca_jan,
  
  "pm25_california_jan2024.csv",
  
  row.names = FALSE
)





pm25_merge <- pm25 %>%
  
  clean_names() %>%
  
  filter(
    state_name == "California"
  )

pm25_merge$date_local <- as.Date(
  pm25_merge$date_local
)

pm25_merge <- pm25_merge %>%
  
  filter(
    
    date_local >= as.Date("2024-01-01") &
      
      date_local <= as.Date("2024-01-31")
  ) %>%
  
  group_by(
    
    county_name,
    
    date_local
  ) %>%
  
  summarise(
    
    pm25 = mean(
      arithmetic_mean,
      na.rm=TRUE
    ),
    
    .groups="drop"
  )





pm25_merge <- pm25 %>%
  
  clean_names() %>%
  
  filter(
    state_name == "California"
  )

pm25_merge$date_local <- as.Date(
  pm25_merge$date_local
)

pm25_merge <- pm25_merge %>%
  
  filter(
    
    date_local >= as.Date("2024-01-01") &
      
      date_local <= as.Date("2024-01-31")
  ) %>%
  
  group_by(
    
    county_name,
    
    date_local
  ) %>%
  
  summarise(
    
    pm25 = mean(
      arithmetic_mean,
      na.rm=TRUE
    ),
    
    .groups="drop"
  )


wind_merge <- wind %>%
  
  clean_names() %>%
  
  filter(
    state_name == "California"
  )

wind_merge$date_local <- as.Date(
  wind_merge$date_local
)

wind_merge <- wind_merge %>%
  
  filter(
    
    date_local >= as.Date("2024-01-01") &
      
      date_local <= as.Date("2024-01-31")
  ) %>%
  
  group_by(
    
    county_name,
    
    date_local
  ) %>%
  
  summarise(
    
    wind_speed = mean(
      arithmetic_mean,
      na.rm=TRUE
    ),
    
    .groups="drop"
  )



temp_merge <- temp %>%
  
  clean_names() %>%
  
  filter(
    state_name == "California"
  )

temp_merge$date_local <- as.Date(
  temp_merge$date_local
)

temp_merge <- temp_merge %>%
  
  filter(
    
    date_local >= as.Date("2024-01-01") &
      
      date_local <= as.Date("2024-01-31")
  ) %>%
  
  group_by(
    
    county_name,
    
    date_local
  ) %>%
  
  summarise(
    
    temperature = mean(
      arithmetic_mean,
      na.rm=TRUE
    ),
    
    .groups="drop"
  )



merged_data <- pm25_merge %>%
  
  left_join(
    
    wind_merge,
    
    by = c(
      "county_name",
      "date_local"
    )
  ) %>%
  
  left_join(
    
    temp_merge,
    
    by = c(
      "county_name",
      "date_local"
    )
  )


head(merged_data)

summary(merged_data)

nrow(merged_data)


merged_data <- merged_data %>%
  
  filter(
    
    !is.na(pm25),
    
    !is.na(wind_speed),
    
    !is.na(temperature)
  )

model1 <- lm(
  
  pm25 ~
    
    wind_speed +
    
    temperature,
  
  data = merged_data
)

summary(model1)


merged_data <- merged_data %>%
  
  arrange(
    county_name,
    date_local
  )

merged_data <- merged_data %>%
  
  group_by(
    county_name
  ) %>%
  
  mutate(
    
    pm25_lag1 = lag(pm25,1),
    
    pm25_lag2 = lag(pm25,2)
  )



model2 <- lm(
  
  pm25 ~
    
    pm25_lag1 +
    
    pm25_lag2 +
    
    wind_speed +
    
    temperature +
    
    wind_speed:temperature,
  
  data = merged_data
)

summary(model2)


