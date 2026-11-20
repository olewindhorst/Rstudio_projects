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

# Reference
reference_model <- lm(units_sold ~ price_eur + discount_pct, data = df)

optimal_theta <- setNames(as.numeric(solve(crossprod(X), crossprod(X, y))), colnames(X))
optimal_cost <- mse_cost(X, y, optimal_theta)

hessian_eigenvalues <- eigen(crossprod(X) / nrow(X))$values
max_stable_learning_rate <- 2 / max(hessian_eigenvalues)

# Learning rates
learning_rates <- c(0.01, 0.1, 0.5, 1.2)

learning_rate_runs <- bind_rows(lapply(learning_rates, function(rate) {
  gradient_descent(X, y, learning_rate = rate, epochs = 200) |>
    mutate(learning_rate = rate)
}))

learning_rate_summary <- learning_rate_runs |>
  group_by(learning_rate) |>
  summarise(
    status = run_status(cost),
    final_cost = last(cost),
    cost_gap = final_cost - optimal_cost,
    epochs_to_1pct = suppressWarnings(min(epoch[cost <= optimal_cost * 1.01])),
    .groups = "drop"
  ) |>
  mutate(
    epochs_to_1pct = ifelse(is.finite(epochs_to_1pct), epochs_to_1pct, NA),
    stable_by_theory = learning_rate < max_stable_learning_rate
  )

# Coefficients
best_run <- learning_rate_runs |>
  filter(learning_rate == 0.5, epoch == max(epoch))

gradient_theta <- unlist(best_run[colnames(X)])

coefficient_check <- tibble(
  term = c("intercept", features),
  gradient_descent = unscale_theta(gradient_theta, scaled$center, scaled$spread),
  lm = unname(coef(reference_model)),
  difference = gradient_descent - lm
)

# Cost surface
slope_grid <- expand_grid(
  price_eur = seq(-40, 10, length.out = 120),
  discount_pct = seq(-10, 40, length.out = 120)
)

cost_surface <- slope_grid |>
  mutate(
    cost = map2_dbl(price_eur, discount_pct, function(price, discount) {
      mse_cost(X, y, c(optimal_theta["intercept"], price, discount))
    })
  )

parameter_path <- learning_rate_runs |>
  filter(learning_rate == 0.1)

# Plots
dir.create("plots", showWarnings = FALSE)

cost_plot <- ggplot(learning_rate_runs, aes(x = epoch, y = cost)) +
  geom_hline(yintercept = optimal_cost, colour = "#263238", linetype = "dashed", linewidth = 0.4) +
  geom_line(colour = "#315b7d", linewidth = 0.7) +
  facet_wrap(vars(learning_rate), scales = "free_y", labeller = label_both) +
  scale_y_log10() +
  theme_minimal(base_size = 12) +
  labs(x = "Epoch", y = "Cost (MSE / 2, log scale)") +
  theme(panel.grid.minor = element_blank())

path_plot <- ggplot(cost_surface, aes(x = price_eur, y = discount_pct)) +
  geom_contour(aes(z = cost), bins = 25, colour = "#d9e4f2") +
  geom_path(data = parameter_path, colour = "#263238", linewidth = 0.5) +
  geom_point(data = parameter_path, colour = "#315b7d", size = 1.4) +
  annotate("point", x = optimal_theta["price_eur"], y = optimal_theta["discount_pct"], colour = "#263238", shape = 4, size = 3) +
  coord_equal() +
  theme_minimal(base_size = 12) +
  labs(x = "Weight price (scaled)", y = "Weight discount (scaled)") +
  theme(panel.grid.minor = element_blank())

ggsave("plots/cost_by_learning_rate.png", cost_plot, width = 8, height = 5, dpi = 150)
ggsave("plots/parameter_path.png", path_plot, width = 7, height = 4.5, dpi = 150)

print(learning_rate_summary)
print(coefficient_check)
cat("Max stable learning rate:", round(max_stable_learning_rate, 4), "\n")
