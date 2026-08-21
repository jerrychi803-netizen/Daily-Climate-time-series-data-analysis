# ==============================================================================
# Complete ETS Training and Evaluation Script for Delhi Climate Dataset
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

# Clean train set: remove 'Grand Total' and any NA values
train_clean <- train_raw %>%
  filter(`Row.Labels` != "Grand Total" & !is.na(`Sum.of.meantemp`))

# Format date columns
train_clean$date <- as.Date(train_clean$Row.Labels, format = "%d/%m/%Y")
test_raw$date    <- as.Date(test_raw$date, format = "%Y-%m-%d")

# Sort chronologically
train_clean <- train_clean %>% arrange(date)
test_raw    <- test_raw %>% arrange(date)

# ------------------------------------------------------------------------------
# 3. Create Time Series Object & Train ETS Model
# ------------------------------------------------------------------------------
# Note: Standard ETS in R requires integer seasonality (frequency <= 24).
# For daily data, setting frequency = 7 captures weekly dynamics, or non-seasonal ETS ("ZNN"/"AAN").
train_ts <- ts(train_clean$Sum.of.meantemp, frequency = 7)

cat("Fitting ETS model (Error, Trend, Seasonal)...\n")
# "ZZZ" automatically selects best Error, Trend, and Seasonal components via AICc
ets_model <- ets(train_ts, model = "ZZZ", damped = TRUE)

# Print fitted model summary
cat("\n--- Fitted ETS Model Summary ---\n")
summary(ets_model)

# ------------------------------------------------------------------------------
# 4. Forecast on Test Set Horizon
# ------------------------------------------------------------------------------
h_steps <- nrow(test_raw)
ets_forecast <- forecast(ets_model, h = h_steps, level = c(80, 95))

# ------------------------------------------------------------------------------
# 5. Out-of-Sample Performance Evaluation
# ------------------------------------------------------------------------------
actual_values    <- test_raw$meantemp
predicted_values <- as.numeric(ets_forecast$mean)

# Calculate accuracy metrics
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
  Lower_95  = as.numeric(ets_forecast$lower[, 2]),
  Upper_95  = as.numeric(ets_forecast$upper[, 2])
)

# Forecast vs Actual Plot
p1 <- ggplot(results_df, aes(x = Date)) +
  geom_ribbon(aes(ymin = Lower_95, ymax = Upper_95), fill = "skyblue", alpha = 0.3) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Forecast, color = "Forecast (ETS)"), size = 1, linetype = "dashed") +
  scale_color_manual(values = c("Actual" = "black", "Forecast (ETS)" = "darkorange")) +
  labs(
    title = "ETS Model: Out-of-Sample Forecast vs Actual",
    subtitle = "Delhi Daily Mean Temperature (Test Period: Jan 2017 - Apr 2017)",
    x = "Date",
    y = "Mean Temperature (°C)",
    color = "Legend"
  ) +
  theme_minimal() +
  theme(legend.position = "top")

print(p1)

# Residual Diagnostic Plot
checkresiduals(ets_model)