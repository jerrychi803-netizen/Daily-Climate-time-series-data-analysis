# ==============================================================================
# Complete ARIMA Training and Evaluation Script for Delhi Climate Dataset
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
# Read CSV files
train_raw <- read.csv("DailyDelhiClimateTrain.csv", stringsAsFactors = FALSE)
test_raw  <- read.csv("DailyDelhiClimateTest.csv", stringsAsFactors = FALSE)

# Clean train set: Remove summary rows (e.g., 'Grand Total')
train_clean <- train_raw %>%
  filter(`Row.Labels` != "Grand Total" & !is.na(`Sum.of.meantemp`))

# Format dates
train_clean$date <- as.Date(train_clean$Row.Labels, format = "%d/%m/%Y")
test_raw$date    <- as.Date(test_raw$date, format = "%Y-%m-%d")

# Sort chronologically
train_clean <- train_clean %>% arrange(date)
test_raw    <- test_raw %>% arrange(date)

cat(sprintf("Training samples: %d (from %s to %s)\n", 
            nrow(train_clean), min(train_clean$date), max(train_clean$date)))
cat(sprintf("Testing samples:  %d (from %s to %s)\n\n", 
            nrow(test_raw), min(test_raw$date), max(test_raw$date)))

# ------------------------------------------------------------------------------
# 3. Create Time Series Object & Train ARIMA
# ------------------------------------------------------------------------------
# Using an annual seasonality frequency (365.25 days/year)
# Note: For strict ARIMA where annual frequency order can be heavy,
# auto.arima uses approximation/stepwise search.
train_ts <- ts(train_clean$Sum.of.meantemp, 
               start = c(2013, 1), 
               frequency = 365.25)

cat("Fitting Auto-ARIMA model...\n")
arima_model <- auto.arima(
  train_ts,
  seasonal = TRUE,
  stepwise = TRUE,
  approximation = FALSE,
  trace = TRUE
)

# Print Model Summary
cat("\n--- Fitted Model Summary ---\n")
summary(arima_model)

# ------------------------------------------------------------------------------
# 4. Forecast on Test Set Horizon
# ------------------------------------------------------------------------------
h_steps <- nrow(test_raw)
arima_forecast <- forecast(arima_model, h = h_steps, level = c(80, 95))

# ------------------------------------------------------------------------------
# 5. Model Evaluation (Out-of-Sample Accuracy)
# ------------------------------------------------------------------------------
actual_values    <- test_raw$meantemp
predicted_values <- as.numeric(arima_forecast$mean)

# Performance Metrics Calculation
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
# Prepare plotting data frame
results_df <- data.frame(
  Date      = test_raw$date,
  Actual    = actual_values,
  Forecast  = predicted_values,
  Lower_95  = as.numeric(arima_forecast$lower[, 2]),
  Upper_95  = as.numeric(arima_forecast$upper[, 2])
)

# Forecast vs Actual Plot
p1 <- ggplot(results_df, aes(x = Date)) +
  geom_ribbon(aes(ymin = Lower_95, ymax = Upper_95), fill = "skyblue", alpha = 0.3) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Forecast, color = "Forecast (ARIMA)"), size = 1, linetype = "dashed") +
  scale_color_manual(values = c("Actual" = "black", "Forecast (ARIMA)" = "firebrick")) +
  labs(
    title = "ARIMA Model: Out-of-Sample Forecast vs Actual",
    subtitle = "Delhi Daily Mean Temperature (Test Period: Jan 2017 - Apr 2017)",
    x = "Date",
    y = "Mean Temperature (°C)",
    color = "Legend"
  ) +
  theme_minimal() +
  theme(legend.position = "top")

print(p1)

# Residual Diagnostic Plot
checkresiduals(arima_model)