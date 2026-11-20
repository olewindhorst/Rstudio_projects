# Helpers
scale_features <- function(x) {
  center <- colMeans(x)
  spread <- apply(x, 2, sd)

  list(
    x = sweep(sweep(x, 2, center), 2, spread, "/"),
    center = center,
    spread = spread
  )
}

design_matrix <- function(x) {
  cbind(intercept = 1, x)
}

mse_cost <- function(X, y, theta) {
  residual <- X %*% theta - y
  sum(residual^2) / (2 * length(y))
}

mse_gradient <- function(X, y, theta) {
  residual <- X %*% theta - y
  as.numeric(crossprod(X, residual)) / length(y)
}

gradient_descent <- function(X, y, learning_rate, epochs, batch_size = nrow(X), seed = 123) {
  set.seed(seed)
  theta <- setNames(rep(0, ncol(X)), colnames(X))
  updates <- 0

  record <- function(epoch) {
    bind_cols(
      tibble(epoch = epoch, updates = updates, cost = mse_cost(X, y, theta)),
      as_tibble_row(theta)
    )
  }

  history <- vector("list", epochs + 1)
  history[[1]] <- record(0)

  for (epoch in seq_len(epochs)) {
    row_order <- if (batch_size < nrow(X)) sample(nrow(X)) else seq_len(nrow(X))
    batches <- split(row_order, ceiling(seq_along(row_order) / batch_size))

    for (rows in batches) {
      theta <- theta - learning_rate * mse_gradient(X[rows, , drop = FALSE], y[rows], theta)
      updates <- updates + 1
    }

    history[[epoch + 1]] <- record(epoch)
  }

  bind_rows(history)
}

unscale_theta <- function(theta, center, spread) {
  slopes <- theta[names(spread)] / spread

  c(
    intercept = unname(theta["intercept"] - sum(slopes * center)),
    slopes
  )
}

run_status <- function(cost, tolerance = 1e-6) {
  last_change <- abs(diff(tail(cost, 2))) / tail(cost, 1)

  case_when(
    !is.finite(tail(cost, 1)) | tail(cost, 1) > cost[1] ~ "diverged",
    last_change < tolerance ~ "converged",
    TRUE ~ "not converged"
  )
}
