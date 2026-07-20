# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Load Dataset
# ----------------------------
# Download Salary_Data.csv from Kaggle or use your own dataset
df = pd.read_csv("Salary_Data.csv")

print("Dataset")
print(df.head())

# ----------------------------
# Select Independent and Dependent Variables
# ----------------------------
X = df[['YearsExperience']]
y = df['Salary']

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Build Linear Regression Model
# ----------------------------
model = LinearRegression()

model.fit(X_train, y_train)

# ----------------------------
# Model Parameters
# ----------------------------
slope = model.coef_[0]
intercept = model.intercept_

print("\nSlope:", slope)
print("Intercept:", intercept)

# ----------------------------
# Prediction
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# Performance Metrics
# ----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nPerformance Metrics")
print("------------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ----------------------------
# Best Fitting Line Equation
# ----------------------------
print("\nBest Fitting Line Equation")
print(f"Salary = {slope:.2f} * YearsExperience + {intercept:.2f}")

# ----------------------------
# Plot Regression Line
# ----------------------------
plt.figure(figsize=(8,5))

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.legend()

plt.show()