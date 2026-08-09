 

library(tidyverse)
library(gridExtra)

set.seed(123)

 

T_true <- as.matrix(
  read.csv("T_true.csv")
)

T_pinn <- as.matrix(
  read.csv("T_pinn.csv")
)

T_nn <- as.matrix(
  read.csv("T_nn.csv")
)

T_gaussian <- as.matrix(
  read.csv("T_gaussian.csv")
)

 

field_metrics <- read.csv(
  "field_metrics.csv"
)

operator_metrics <- read.csv(
  "operator_metrics.csv"
)

spillover_results <- read.csv(
  "spillover_results.csv"
)

print(field_metrics)
print(operator_metrics)
print(spillover_results)
 
