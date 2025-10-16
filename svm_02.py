import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("fitness_dataset.csv")

# Handle missing values
df['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].mean())

# Normalize smokes column (map yes/no/0 to binary)
df['smokes'] = df['smokes'].replace({'yes': 1, 'no': 0, '0': 0}).astype(int)

# Encode gender
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])  # M/F → 0/1

# Features and target
X = df.drop(columns=['is_fit'])
y = df['is_fit']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
svm_clf = SVC(kernel='rbf', random_state=42)
svm_clf.fit(X_train_scaled, y_train)

# Predictions
y_pred = svm_clf.predict(X_test_scaled)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Not Fit', 'Fit'], yticklabels=['Not Fit', 'Fit'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot Accuracy Comparison
plt.figure(figsize=(5,4))
sns.barplot(x=["Accuracy"], y=[acc], color="skyblue")
plt.ylim(0,1)
plt.title("SVM Model Accuracy")
plt.show()
