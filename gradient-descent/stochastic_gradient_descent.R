# Libs
library(tidyverse)
library(dplyr)
library(readr)

source("helpers.R")

# Data
df <- read.csv("data/demo_data.csv")

features <- c("price_eur", "discount_pct")

scaled <- scale_features(as.matrix(df[features]))
X <- design_matrix(scaled$x)
y <- df$units_sold

optimal_theta <- setNames(as.numeric(solve(crossprod(X), crossprod(X, y))), colnames(X))
optimal_cost <- mse_cost(X, y, optimal_theta)

# Variants
variants <- tibble(
  method = c("Batch", "Mini-batch", "Stochastic"),
  batch_size = c(nrow(X), 16, 1)
)

variant_runs <- bind_rows(lapply(seq_len(nrow(variants)), function(i) {
  gradient_descent(X, y, learning_rate = 0.05, epochs = 50, batch_size = variants$batch_size[i]) |>
    mutate(method = variants$method[i], batch_size = variants$batch_size[i])
})) |>
  mutate(method = factor(method, levels = variants$method))

variant_summary <- variant_runs |>
  group_by(method, batch_size) |>
  summarise(
    updates = max(updates),
    cost_after_5_epochs = cost[epoch == 5],
    final_cost = last(cost),
    cost_gap = final_cost - optimal_cost,
    late_cost_sd = sd(tail(cost, 10)),
    .groups = "drop"
  )

# Plots
dir.create("plots", showWarnings = FALSE)

variant_plot <- ggplot(variant_runs, aes(x = epoch, y = cost - optimal_cost)) +
  geom_line(colour = "#315b7d", linewidth = 0.7) +
  facet_wrap(vars(method)) +
  scale_y_log10() +
  theme_minimal(base_size = 12) +
  labs(x = "Epoch", y = "Cost above optimum (log scale)") +
  theme(panel.grid.minor = element_blank())

ggsave("plots/batch_size_comparison.png", variant_plot, width = 8, height = 4.5, dpi = 150)

print(variant_summary)
