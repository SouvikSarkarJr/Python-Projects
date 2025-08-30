
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- Load Dataset ----------------clea
file_path = "Car_Price_Prediction.csv"
df = pd.read_csv(file_path)

# ---------------- Features & Target ----------------
X = df.drop("Price", axis=1)
y = df["Price"]

categorical_cols = ["Make", "Model", "Fuel Type", "Transmission"]
numeric_cols = ["Year", "Engine Size", "Mileage"]

# Preprocessor for encoding categoricals
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Linear Regression ----------------
lin_model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

print("\n--- Linear Regression ---")
print("R2 Score:", r2_score(y_test, y_pred_lin))
print("MSE:", mean_squared_error(y_test, y_pred_lin))

# ---------------- Polynomial Regression (numeric only) ----------------
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

print("\n--- Polynomial Regression (degree=2) ---")
print("R2 Score:", r2_score(y_test, y_pred_poly))
print("MSE:", mean_squared_error(y_test, y_pred_poly))

# ---------------- Ridge Regression ----------------
ridge_model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", Ridge(alpha=1.0))
])
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

print("\n--- Ridge Regression ---")
print("R2 Score:", r2_score(y_test, y_pred_ridge))

# ---------------- Lasso Regression ----------------
lasso_model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", Lasso(alpha=0.1, max_iter=10000))
])
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

print("\n--- Lasso Regression ---")
print("R2 Score:", r2_score(y_test, y_pred_lasso))