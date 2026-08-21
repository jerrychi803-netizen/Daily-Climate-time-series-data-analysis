# ==============================================================================
# Complete Random Forest Training and Evaluation Script for Delhi Climate Dataset
# ==============================================================================

# 1. Install & Load Required Packages
required_packages <- c("ranger", "ggplot2", "dplyr", "lubridate")
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) install.packages(new_packages)

library(ranger)
library(ggplot2)
library(dplyr)
library(lubridate)

# ------------------------------------------------------------------------------
# 2. Load and Preprocess Data
# ------------------------------------------------------------------------------
train_raw <- read.csv("DailyDelhiClimateTrain.csv", stringsAsFactors = FALSE)
test_raw  <- read.csv("DailyDelhiClimateTest.csv", stringsAsFactors = FALSE)

# Clean train set: filter out summary rows ('Grand Total') and missing target values
train_clean <- train_raw %>%
  filter(`Row.Labels` != "Grand Total" & !is.na(`Sum.of.meantemp`)) %>%
  rename(
    date = Row.Labels,
    meantemp = Sum.of.meantemp,
    humidity = Sum.of.humidity,
    wind_speed = Sum.of.wind_speed
  )

train_clean$date <- as.Date(train_clean$date, format = "%d/%m/%Y")
test_raw$date    <- as.Date(test_raw$date, format = "%Y-%m-%d")

# Combine datasets chronologically to create seamless lag/rolling features
train_clean$split <- "train"
test_raw$split    <- "test"

# Align columns
common_cols <- c("date", "meantemp", "humidity", "wind_speed", "split")
combined_df <- rbind(train_clean[, common_cols], test_raw[, common_cols]) %>%
  arrange(date)

# ------------------------------------------------------------------------------
# 3. Feature Engineering (Lags, Moving Averages, Calendar Variables)
# ------------------------------------------------------------------------------
combined_features <- combined_df %>%
  mutate(
    # Calendar & cyclical features
    day_of_year = yday(date),
    month       = month(date),
    day_of_week = wday(date),
    
    # Harmonic seasonal components (sine & cosine of annual cycle)
    sin_year = sin(2 * pi * day_of_year / 365.25),
    cos_year = cos(2 * pi * day_of_year / 365.25),
    
    # Lag features
    lag_1  = lag(meantemp, 1),
    lag_2  = lag(meantemp, 2),
    lag_7  = lag(meantemp, 7),
    lag_14 = lag(meantemp, 14),
    lag_30 = lag(meantemp, 30),
    
    # Exogenous lags
    humidity_lag1   = lag(humidity, 1),
    wind_speed_lag1 = lag(wind_speed, 1),
    
    # Rolling aggregations (prior 7 days)
    roll_mean_7 = stats::filter(lag(meantemp, 1), rep(1/7, 7), sides = 1)
  )

# Separate back into train and test sets
train_set <- combined_features %>% filter(split == "train") %>% na.omit()
test_set  <- combined_features %>% filter(split == "test")  %>% na.omit()

# Define predictor variables
feature_vars <- c("day_of_year", "month", "day_of_week", "sin_year", "cos_year",
                  "lag_1", "lag_2", "lag_7", "lag_14", "lag_30",
                  "humidity_lag1", "wind_speed_lag1", "roll_mean_7")

rf_formula <- as.formula(paste("meantemp ~", paste(feature_vars, collapse = " + ")))

# ------------------------------------------------------------------------------
# 4. Train Random Forest Model
# ------------------------------------------------------------------------------
cat("Training Random Forest model...\n")
rf_model <- ranger(
  formula       = rf_formula,
  data          = train_set,
  num.trees     = 500,
  mtry          = floor(sqrt(length(feature_vars))),
  importance    = "impurity",
  seed          = 42
)

# Print model summary
cat("\n--- Fitted Random Forest Summary ---\n")
print(rf_model)

# ------------------------------------------------------------------------------
# 5. Predict & Out-of-Sample Evaluation
# ------------------------------------------------------------------------------
rf_pred <- predict(rf_model, data = test_set)$predictions

actual_values    <- test_set$meantemp
predicted_values <- rf_pred

# Calculate accuracy metrics
mae  <- mean(abs(actual_values - predicted_values))
rmse <- sqrt(mean((actual_values - predicted_values)^2))
mape <- mean(abs((actual_values - predicted_values) / actual_values)) * 100

cat("\n--- Test Set Performance Metrics ---\n")
cat(sprintf("MAE  : %.4f °C\n", mae))
cat(sprintf("RMSE : %.4f °C\n", rmse))
cat(sprintf("MAPE : %.2f %%\n\n", mape))

# ------------------------------------------------------------------------------
# 6. Visualizations
# ------------------------------------------------------------------------------
# 1. Forecast vs Actual
results_df <- data.frame(
  Date     = test_set$date,
  Actual   = actual_values,
  Forecast = predicted_values
)

p1 <- ggplot(results_df, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Forecast, color = "Random Forest"), size = 1, linetype = "dashed") +
  scale_color_manual(values = c("Actual" = "black", "Random Forest" = "forestgreen")) +
  labs(
    title = "Random Forest Model: Out-of-Sample Prediction vs Actual",
    subtitle = "Delhi Daily Mean Temperature (Test Period: Jan 2017 - Apr 2017)",
    x = "Date",
    y = "Mean Temperature (°C)",
    color = "Legend"
  ) +
  theme_minimal() +
  theme(legend.position = "top")

print(p1)

# 2. Variable Importance Plot
importance_df <- data.frame(
  Feature    = names(rf_model$variable.importance),
  Importance = rf_model$variable.importance
) %>% arrange(desc(Importance))

p2 <- ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "forestgreen") +
  coord_flip() +
  labs(
    title = "Random Forest: Feature Importance",
    x = "Features",
    y = "Node Impurity Reduction"
  ) +
  theme_minimal()

print(p2)