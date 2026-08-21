# ==============================================================================
# Complete TBATS Training and Evaluation Script for Delhi Climate Dataset
# ==============================================================================

# 1. Install & Load Required Packages
required_packages <- c("forecast", "ggplot2", "dplyr")
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) install.packages(new_packages)

library(forecast)
library(ggplot2)
library(dplyr)

# ------------------------------------------------------------------------------
# 2. Load and Preprocess Data
# ------------------------------------------------------------------------------
train_raw <- read.csv("DailyDelhiClimateTrain.csv", stringsAsFactors = FALSE)
test_raw  <- read.csv("DailyDelhiClimateTest.csv", stringsAsFactors = FALSE)

# Clean train set: filter out aggregate rows ('Grand Total') and NAs
train_clean <- train_raw %>%
  filter(`Row.Labels` != "Grand Total" & !is.na(`Sum.of.meantemp`))

# Format date columns
train_clean$date <- as.Date(train_clean$Row.Labels, format = "%d/%m/%Y")
test_raw$date    <- as.Date(test_raw$date, format = "%Y-%m-%d")

# Sort chronologically
train_clean <- train_clean %>% arrange(date)
test_raw    <- test_raw %>% arrange(date)

# ------------------------------------------------------------------------------
# 3. Create Time Series Object & Train TBATS Model
# ------------------------------------------------------------------------------
# TBATS natively handles non-integer, long seasonal periods (365.25 days)
# via trigonometric Fourier terms and Box-Cox transformations.
train_ts <- ts(train_clean$Sum.of.meantemp, 
               start = c(2013, 1), 
               frequency = 365.25)

cat("Fitting TBATS model (this may take a minute)...\n")
tbats_model <- tbats(
  train_ts,
  use.box.cox = TRUE,
  use.trend = TRUE,
  use.damped.trend = TRUE,
  seasonal.periods = c(365.25)
)

# Print fitted model details
cat("\n--- Fitted TBATS Model Summary ---\n")
print(tbats_model)

# ------------------------------------------------------------------------------
# 4. Forecast on Test Horizon
# ------------------------------------------------------------------------------
h_steps <- nrow(test_raw)
tbats_forecast <- forecast(tbats_model, h = h_steps, level = c(80, 95))

# ------------------------------------------------------------------------------
# 5. Out-of-Sample Performance Evaluation
# ------------------------------------------------------------------------------
actual_values    <- test_raw$meantemp
predicted_values <- as.numeric(tbats_forecast$mean)

# Performance metrics calculation
mae  <- mean(abs(actual_values - predicted_values))
rmse <- sqrt(mean((actual_values - predicted_values)^2))
mape <- mean(abs((actual_values - predicted_values) / actual_values)) * 100

cat("\n--- Test Set Performance Metrics ---\n")
cat(sprintf("MAE  : %.4f °C\n", mae))
cat(sprintf("RMSE : %.4f °C\n", rmse))
cat(sprintf("MAPE : %.2f %%\n\n", mape))

# ------------------------------------------------------------------------------
# 6. Visualization
# ------------------------------------------------------------------------------
results_df <- data.frame(
  Date      = test_raw$date,
  Actual    = actual_values,
  Forecast  = predicted_values,
  Lower_95  = as.numeric(tbats_forecast$lower[, 2]),
  Upper_95  = as.numeric(tbats_forecast$upper[, 2])
)

# Forecast vs Actual Plot
p1 <- ggplot(results_df, aes(x = Date)) +
  geom_ribbon(aes(ymin = Lower_95, ymax = Upper_95), fill = "mediumpurple1", alpha = 0.3) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Forecast, color = "Forecast (TBATS)"), size = 1, linetype = "dashed") +
  scale_color_manual(values = c("Actual" = "black", "Forecast (TBATS)" = "purple4")) +
  labs(
    title = "TBATS Model: Out-of-Sample Forecast vs Actual",
    subtitle = "Delhi Daily Mean Temperature (Test Period: Jan 2017 - Apr 2017)",
    x = "Date",
    y = "Mean Temperature (°C)",
    color = "Legend"
  ) +
  theme_minimal() +
  theme(legend.position = "top")

print(p1)

# Residual Diagnostic Plot
checkresiduals(tbats_model)