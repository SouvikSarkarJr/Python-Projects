import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load dataset
file_path = "Car_Price_Prediction.csv"   # <--- put your CSV file name here
df = pd.read_csv(file_path)

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Categorical & numerical features
categorical_cols = ["Make", "Model", "Fuel Type", "Transmission"]
numeric_cols = ["Year", "Engine Size", "Mileage"]

# Preprocessor for encoding categoricals
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Polynomial Regression ----------------
poly_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("poly", PolynomialFeatures(degree=2, include_bias=False), numeric_cols)
])

poly_model = Pipeline([
    ("preprocess", poly_preprocessor),
    ("regressor", LinearRegression())
])
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

print("\n--- Non-Linear Regression: Polynomial (degree=2) ---")
print("R2 Score:", r2_score(y_test, y_pred_poly))
print("MSE:", mean_squared_error(y_test, y_pred_poly))

# ---------------- Decision Tree Regression ----------------
dt_model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", DecisionTreeRegressor(max_depth=10, random_state=42))
])
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\n--- Non-Linear Regression: Decision Tree ---")
print("R2 Score:", r2_score(y_test, y_pred_dt))
print("MSE:", mean_squared_error(y_test, y_pred_dt))

# ---------------- Random Forest Regression ----------------
rf_model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n--- Non-Linear Regression: Random Forest ---")
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))

import matplotlib.pyplot as plt

# Plot Actual vs Predicted for all models
plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred_poly, alpha=0.6, color="red", label="Polynomial (deg=2)")
plt.scatter(y_test, y_pred_dt, alpha=0.6, color="green", label="Decision Tree")
plt.scatter(y_test, y_pred_rf, alpha=0.6, color="blue", label="Random Forest")

# Ideal line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2, label="Ideal Fit")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Non-Linear Regression Models Comparison")
plt.legend()
plt.grid(True)
plt.show()