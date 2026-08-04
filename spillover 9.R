ggsave(
  
  "figure_sigma_sensitivity.png",
  
  p6,
  
  width = 7,
  
  height = 5,
  
  dpi = 600
)


# =====================================================
# LIBRARIES
# =====================================================

library(sf)

library(tidyverse)

library(tigris)

library(geosphere)

library(patchwork)

options(
  
  tigris_use_cache = TRUE
)

# =====================================================
# CALIFORNIA COUNTY SHAPEFILE
# =====================================================

ca_counties <- counties(
  
  state = "CA",
  
  cb = TRUE,
  
  class = "sf"
)

# =====================================================
# COUNTY CENTROIDS
# =====================================================

county_centroids <- st_centroid(
  
  ca_counties
)

coords <- st_coordinates(
  
  county_centroids
)

county_centroids$lon <- coords[,1]

county_centroids$lat <- coords[,2]

# =====================================================
# COUNTY NAME CLEANING
# =====================================================

county_centroids$county_name <- county_centroids$NAME

# =====================================================
# AVERAGE PM25
# =====================================================

pm_map <- model_data %>%
  
  group_by(county_name) %>%
  
  summarise(
    
    mean_pm25 =
      mean(pm25, na.rm = TRUE),
    
    mean_spillover =
      mean(spillover, na.rm = TRUE)
  )

# =====================================================
# MERGE
# =====================================================

map_data <- ca_counties %>%
  
  left_join(
    
    pm_map,
    
    by = c(
      "NAME" = "county_name"
    )
  )

# =====================================================
# PANEL A
# COUNTY PM25 MAP
# =====================================================

p_map_pm25 <- ggplot(
  
  map_data
) +
  
  geom_sf(
    
    aes(
      fill = mean_pm25
    ),
    
    color = "white",
    
    linewidth = 0.3
  ) +
  
  scale_fill_viridis_c(
    
    option = "C",
    
    name = expression(
      PM[2.5]
    )
  ) +
  
  labs(
    
    title =
      "County Mean PM2.5"
  ) +
  
  theme_void() +
  
  theme(
    
    plot.title =
      element_text(
        
        size = 18,
        
        face = "bold",
        
        hjust = 0.5
      ),
    
    legend.position =
      "right"
  )

# =====================================================
# PANEL B
# SPILLOVER MAP
# =====================================================

p_map_spill <- ggplot(
  
  map_data
) +
  
  geom_sf(
    
    aes(
      fill = mean_spillover
    ),
    
    color = "white",
    
    linewidth = 0.3
  ) +
  
  scale_fill_viridis_c(
    
    option = "B",
    
    name = "Spillover"
  ) +
  
  labs(
    
    title =
      "Gaussian Spillover Intensity"
  ) +
  
  theme_void() +
  
  theme(
    
    plot.title =
      element_text(
        
        size = 18,
        
        face = "bold",
        
        hjust = 0.5
      ),
    
    legend.position =
      "right"
  )

# =====================================================
# PANEL C
# TRANSPORT NETWORK
# =====================================================

coords_matrix <- county_centroids %>%
  
  st_drop_geometry() %>%
  
  select(
    lon,
    lat
  ) %>%
  
  as.matrix()

dist_matrix <- distm(
  
  coords_matrix,
  
  fun = distHaversine
) / 1000

sigma <- 25

weight_matrix <- exp(
  
  -(dist_matrix^2) /
    
    (2 * sigma^2)
)

diag(weight_matrix) <- 0

# =====================================================
# BUILD EDGES
# =====================================================

edge_list <- data.frame()

for(i in 1:nrow(weight_matrix)){
  
  for(j in 1:ncol(weight_matrix)){
    
    if(weight_matrix[i,j] > 0.20){
      
      edge_list <- rbind(
        
        edge_list,
        
        data.frame(
          
          x =
            coords_matrix[i,1],
          
          y =
            coords_matrix[i,2],
          
          xend =
            coords_matrix[j,1],
          
          yend =
            coords_matrix[j,2],
          
          weight =
            weight_matrix[i,j]
        )
      )
    }
  }
}

# =====================================================
# NETWORK MAP
# =====================================================

p_network <- ggplot() +
  
  geom_sf(
    
    data = ca_counties,
    
    fill = "grey95",
    
    color = "white"
  ) +
  
  geom_segment(
    
    data = edge_list,
    
    aes(
      
      x = x,
      y = y,
      xend = xend,
      yend = yend,
      alpha = weight
    ),
    
    color = "red",
    
    linewidth = 0.5
  ) +
  
  geom_point(
    
    data =
      county_centroids %>%
      st_drop_geometry(),
    
    aes(
      lon,
      lat
    ),
    
    color = "black",
    
    size = 1.8
  ) +
  
  scale_alpha(
    
    range = c(
      0.05,
      0.7
    )
  ) +
  
  labs(
    
    title =
      "Spatial Transport Network"
  ) +
  
  theme_void() +
  
  theme(
    
    plot.title =
      element_text(
        
        size = 18,
        
        face = "bold",
        
        hjust = 0.5
      ),
    
    legend.position =
      "none"
  )

# =====================================================
# COMBINE
# =====================================================

spatial_figure <- (
  
  p_map_pm25 |
    
    p_map_spill |
    
    p_network
  
) +
  
  plot_annotation(
    
    tag_levels = "A"
  )

# =====================================================
# SAVE
# =====================================================

ggsave(
  
  "spatial_transport_figure.png",
  
  spatial_figure,
  
  width = 18,
  
  height = 6,
  
  dpi = 600,
  
  bg = "white"
)

# =====================================================
# DISPLAY
# =====================================================

print(
  spatial_figure
)