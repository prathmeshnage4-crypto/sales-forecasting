import pandas as pd

# Load the sales dataset
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")
# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Shape:", df.shape)

# Display column names
print("\nColumns:")
print(df.columns)

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Sort data by Order Date
df = df.sort_values("Order Date")

# Display date range
print("\nDate Range:")
print(df["Order Date"].min(), "to", df["Order Date"].max())

# Create monthly sales data
monthly_sales = df.resample("ME", on="Order Date")["Sales"].sum()

# Display monthly sales
print("\nMonthly Sales:")
print(monthly_sales.head(12))

import matplotlib.pyplot as plt


# Plot monthly sales
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales.values)

plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sales_trend.png")
plt.show()


# Create lag features for forecasting
forecast_df = monthly_sales.to_frame(name="Sales")

forecast_df["Lag_1"] = forecast_df["Sales"].shift(1)
forecast_df["Lag_2"] = forecast_df["Sales"].shift(2)
forecast_df["Lag_3"] = forecast_df["Sales"].shift(3)

# Remove missing values
forecast_df = forecast_df.dropna()

print("\nForecasting Data:")
print(forecast_df.head())

# Prepare features and target
X = forecast_df[["Lag_1", "Lag_2", "Lag_3"]]
y = forecast_df["Sales"]

# Chronological train-test split
split_point = int(len(forecast_df) * 0.80)

X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]

y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

from sklearn.ensemble import RandomForestRegressor

# Create the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# Make predictions on test data
y_pred = model.predict(X_test)

print("\nActual Sales:")
print(y_test.values)

print("\nPredicted Sales:")
print(y_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)

# Plot actual vs predicted sales
plt.figure(figsize=(12, 6))

plt.plot(y_test.index, y_test.values, label="Actual Sales")
plt.plot(y_test.index, y_pred, label="Predicted Sales")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("actual_vs_predicted_sales.png")
plt.show()