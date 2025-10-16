import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ===== Load Dataset =====
file_path = "framingham.csv"   # dataset you uploaded
df = pd.read_csv(file_path)

print("Dataset Head:")
print(df.head())
print("\nColumns:", df.columns)

# ===== Handle Missing Values =====
df = df.dropna()   # drop rows with missing values

# ===== Select Features =====
# Choose 2 features for visualization
X = df[["age", "totChol"]].values   # features
y = df["TenYearCHD"].values         # target (0 = no disease, 1 = disease)

# ===== Train-Test Split =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== Feature Scaling =====
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===== Train KNN =====
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ===== Predictions =====
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ===== Visualization with Decision Boundary =====
h = 0.1  # step size in mesh

# Create mesh grid
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k', cmap="coolwarm", s=80)
plt.title("KNN Decision Boundary (age vs totChol)")
plt.xlabel("Age (standardized)")
plt.ylabel("Total Cholesterol (standardized)")
plt.show()
