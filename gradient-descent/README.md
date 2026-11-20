# Gradient Descent for a Simple Sales Model

This R project fits a linear regression with gradient descent instead of the closed-form solution of `lm()`.

```text
start weights -> gradient of the cost -> small step downhill -> repeat
```

The scripts show how the learning rate and the batch size change the path to the optimum. The result is checked against `lm()`, so the method can be trusted before it is used on larger problems.

## Data

[`data/demo_data.csv`](data/demo_data.csv) contains 100 synthetic rows created by [`data/birth.R`](data/birth.R).

| Column | Meaning |
| --- | --- |
| `customer_id` | Synthetic customer ID |
| `price_eur` | Unit price between 5 and 20 EUR |
| `discount_pct` | Discount in percent, correlated with the price |
| `units_sold` | Number of pieces sold |

The data is generated from a known relation plus noise:

```text
units_sold = 150 - 5 * price_eur + 2 * discount_pct + noise
```

Price and discount are deliberately correlated (r = 0.70). This makes the cost surface elongated and gradient descent slower, which is the typical situation in real sales data.

## Calculation

The model and its cost:

```text
prediction = intercept + w_price * price + w_discount * discount
cost       = sum((prediction - units_sold)^2) / (2 * n)
```

One gradient descent step:

```text
gradient = t(X) %*% (X %*% theta - y) / n
theta    = theta - learning_rate * gradient
```

Both features are standardised (mean 0, standard deviation 1) before the descent. Without scaling, one learning rate is too large for one weight and too small for the other. The weights are transformed back to EUR and percent at the end.

The largest stable learning rate follows from the curvature of the cost:

```text
max learning rate = 2 / largest eigenvalue of t(X) %*% X / n
```

## Scripts

| Script | Purpose |
| --- | --- |
| [`helpers.R`](helpers.R) | Scaling, cost, gradient and one `gradient_descent()` function for all batch sizes |
| [`gradient_descent.R`](gradient_descent.R) | Batch gradient descent with four learning rates, check against `lm()`, cost surface |
| [`stochastic_gradient_descent.R`](stochastic_gradient_descent.R) | Batch, mini-batch (16 rows) and stochastic (1 row) with the same learning rate |

`gradient_descent()` counts epochs. One epoch is one pass over all rows. Batch gradient descent makes one update per epoch, mini-batch 7 and stochastic 100.

## Run

Install the packages once:

```r
install.packages("tidyverse")
```

Open `gradient-descent` as the working directory and run:

```r
source("gradient_descent.R")
source("stochastic_gradient_descent.R")
```

To recreate the demo data, run `data/birth.R` with `data` as the working directory.

The main results stay available in R as:

- `learning_rate_summary`
- `coefficient_check`
- `learning_rate_runs`
- `variant_summary`
- `variant_runs`

The descent function can be used directly:

```r
gradient_descent(X, y, learning_rate = 0.1, epochs = 100)
gradient_descent(X, y, learning_rate = 0.05, epochs = 50, batch_size = 16)
```

## Plots

The scripts create three minimal plots in the `plots` folder:

- `cost_by_learning_rate.png` – cost per epoch for each learning rate, dashed line = optimum
- `parameter_path.png` – path of both weights over the cost contours, cross = optimum
- `batch_size_comparison.png` – distance to the optimum per epoch for batch, mini-batch and stochastic

No CSV result files are written.

## Demo result

Learning rates, batch gradient descent, 200 epochs:

| Learning rate | Status | Epochs to within 1% of optimum |
| ---: | --- | ---: |
| 0.01 | not converged | – |
| 0.1 | converged | 92 |
| 0.5 | converged | 18 |
| 1.2 | diverged | – |

The theoretical limit is 1.19. A learning rate of 1.2 is just above it and the cost grows again after a few epochs.

Coefficients after 200 epochs with learning rate 0.5:

| Term | Gradient descent | `lm()` |
| --- | ---: | ---: |
| Intercept | 150.9732 | 150.9732 |
| `price_eur` | -4.9080 | -4.9080 |
| `discount_pct` | 1.8201 | 1.8201 |

The difference is below 1e-13. The estimates are close to the true values of -5 and 2 used in `birth.R`.

Batch sizes, learning rate 0.05, 50 epochs:

| Method | Updates | Cost after 5 epochs | Final cost above optimum |
| --- | ---: | ---: | ---: |
| Batch | 50 | 3720.92 | 51.28 |
| Mini-batch | 350 | 217.43 | 0.0024 |
| Stochastic | 5000 | 28.07 | 0.0118 |

Stochastic gradient descent gets close fastest, but keeps jumping around the optimum because every step uses only one row. Mini-batch is the usual compromise.

## Limits

The demo uses a convex cost with one global minimum. For this problem `lm()` is faster and exact. Gradient descent pays off for very large data sets or models without a closed-form solution.

The learning rate is fixed. Learning rate schedules, momentum and early stopping are not part of this demo.
