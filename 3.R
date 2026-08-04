library(dplyr)

old <- read.csv("Environmental_Results.csv")
new <- read.csv("Environmental_Results_Site.csv")

comparison <-
  old %>%
  rename(
    Beta_Original = Estimate,
    SE_Original = Std_Error,
    P_Original = P_Value
  ) %>%
  left_join(
    new %>%
      rename(
        Beta_Site = Estimate,
        SE_Site = Std_Error,
        P_Site = P_Value
      ),
    by=c("Outcome","Exposure")
  ) %>%
  mutate(
    Percent_Change =
      100*(Beta_Site-Beta_Original)/abs(Beta_Original)
  )

write.csv(
  comparison,
  "Site_Adjustment_Comparison.csv",
  row.names=FALSE
)