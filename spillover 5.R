# =====================================================
# EXPORT EPA REAL DATA
# =====================================================

export_data <- model_data %>%
  
  select(
    
    county_name,
    
    date_local,
    
    pm25,
    
    pm25_lag1,
    
    pm25_lag2,
    
    spillover,
    
    wind_speed,
    
    temperature
  )

write.csv(
  
  export_data,
  
  "epa_pm25_transport.csv",
  
  row.names = FALSE
)

cat(
  
  "\nEPA REAL DATA EXPORTED\n"
)


# =====================================================
# EXPORT COUNTY COORDINATES
# =====================================================

county_coordinates <- county_coords %>%
  
  select(
    
    county_name,
    
    latitude,
    
    longitude
  )

write.csv(
  
  county_coordinates,
  
  "county_coordinates.csv",
  
  row.names = FALSE
)

cat(
  
  "\ncounty_coordinates.csv EXPORTED\n"
)