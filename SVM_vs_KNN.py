# ===============================
# 📊 Fitness Dataset: SVM vs KNN
# ===============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Load Dataset ---
df = pd.read_csv("fitness_dataset.csv")

# --- Handle Missing Values ---
imputer = SimpleImputer(strategy='mean')
df['sleep_hours'] = imputer.fit_transform(df[['sleep_hours']])

# --- Encode Categorical Columns ---
le_smokes = LabelEncoder()
le_gender = LabelEncoder()
df['smokes'] = le_smokes.fit_transform(df['smokes'].astype(str))
df['gender'] = le_gender.fit_transform(df['gender'])

# --- Split Features and Target ---
X = df.drop('is_fit', axis=1)
y = df['is_fit']

# --- Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- Train SVM Model ---
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# --- Train KNN Model ---
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# --- Evaluate Models ---
def evaluate_model(name, y_true, y_pred):
    print(f"\n===== {name} Evaluation =====")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

evaluate_model("SVM", y_test, y_pred_svm)
evaluate_model("KNN", y_test, y_pred_knn)

# --- Summary Comparison ---
acc_svm = accuracy_score(y_test, y_pred_svm)
acc_knn = accuracy_score(y_test, y_pred_knn)
print("\n==============================")
print("📈 Model Comparison Summary")
print("==============================")
print(f"SVM Accuracy: {acc_svm:.4f}")
print(f"KNN Accuracy: {acc_knn:.4f}")

if acc_svm > acc_knn:
    print("✅ SVM performs better overall.")
else:
    print("✅ KNN performs better overall.")
