import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

# 1. Generate Dummy Data
np.random.seed(42)

# Time periods
time_periods = 100

# Generate synthetic marketing activities and sales data
tv_spend = np.random.uniform(5000, 20000, size=time_periods)
digital_spend = np.random.uniform(2000, 15000, size=time_periods)
promotions = np.random.uniform(1000, 10000, size=time_periods)
base_sales = np.random.uniform(20000, 50000, size=time_periods)

# Ad-stock transformation (decay)
def adstock_transform(spend, rate=0.5, lag=3):
    result = np.zeros_like(spend)
    for t in range(len(spend)):
        if t == 0:
            result[t] = spend[t]
        else:
            result[t] = spend[t] + rate * result[t - 1]
    return result

# Apply ad-stock transformations
tv_adstock = adstock_transform(tv_spend, rate=0.5)
digital_adstock = adstock_transform(digital_spend, rate=0.3)
promotion_adstock = adstock_transform(promotions, rate=0.7)

# Diminishing returns function
def diminishing_returns(x, alpha, beta):
    return alpha * (1 - np.exp(-beta * x))

# Apply diminishing returns
tv_effective = diminishing_returns(tv_adstock, alpha=0.5, beta=0.001)
digital_effective = diminishing_returns(digital_adstock, alpha=0.7, beta=0.002)
promotion_effective = diminishing_returns(promotion_adstock, alpha=0.6, beta=0.003)

# Generate total sales with random noise
sales = (
    base_sales
    + 0.3 * tv_effective
    + 0.4 * digital_effective
    + 0.2 * promotion_effective
    + np.random.normal(scale=5000, size=time_periods)
)

# Create a DataFrame
data = pd.DataFrame({
    "Sales": sales,
    "TV_Spend": tv_spend,
    "Digital_Spend": digital_spend,
    "Promotions": promotions,
    "TV_Adstock": tv_adstock,
    "Digital_Adstock": digital_adstock,
    "Promotion_Adstock": promotion_adstock
})

# 2. Fit a Mixed Effect Regression Model
X = data[["TV_Adstock", "Digital_Adstock", "Promotion_Adstock"]]
y = data["Sales"]

model = LinearRegression()
model.fit(X, y)

# Model evaluation
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

# 3. Results and Visualization
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-Squared: {r2:.2f}")
print("\nModel Coefficients:")
print(coefficients)

# Plot actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.plot(data["Sales"], label="Actual Sales")
plt.plot(y_pred, label="Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Time Period")
plt.ylabel("Sales")
plt.legend()
plt.show()

# 4. ROI Calculation
tv_roi = coefficients.loc[coefficients["Feature"] == "TV_Adstock", "Coefficient"].values[0] / tv_spend.mean()
digital_roi = coefficients.loc[coefficients["Feature"] == "Digital_Adstock", "Coefficient"].values[0] / digital_spend.mean()
promotion_roi = coefficients.loc[coefficients["Feature"] == "Promotion_Adstock", "Coefficient"].values[0] / promotions.mean()

print("\nEstimated ROI per Channel:")
print(f"TV ROI: {tv_roi:.5f}")
print(f"Digital ROI: {digital_roi:.5f}")
print(f"Promotions ROI: {promotion_roi:.5f}")
