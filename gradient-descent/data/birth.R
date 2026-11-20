# Libs
library(tidyverse)
library(dplyr)
library(readr)

# Seed
set.seed(123)

# Data
df <- data.frame(
  customer_id = paste0("id_", 1:100),
  price_eur = round(runif(100, min = 5, max = 20), 2)
)

df$discount_pct <- round( #Higher prices get higher discounts, so both features are correlated
  pmin(pmax(0.9 * df$price_eur + rnorm(100, mean = 0, sd = 4), 0), 30),
  1
)

df$units_sold <- round( #True relation: -5 units per EUR, +2 units per discount point
  150 - 5 * df$price_eur + 2 * df$discount_pct + rnorm(100, mean = 0, sd = 8)
)

# Save
write.csv(df, "demo_data.csv", row.names = FALSE)
