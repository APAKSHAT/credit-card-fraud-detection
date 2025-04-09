# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv('creditcard.csv')  # Replace with your path

# Check class distribution
print("Class Distribution:\n", df['Class'].value_counts())

# Features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Anomaly Detection (Optional - Isolation Forest)
iso_forest = IsolationForest(contamination=0.001, random_state=42)
anomaly_preds = iso_forest.fit_predict(X_scaled)

# Keep only non-anomalous data (optional step)
X_clean = X_scaled[anomaly_preds == 1]
y_clean = y[anomaly_preds == 1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean)

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("\n🎯 Random Forest Results:")
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))
print("Accuracy:", accuracy_score(y_test, rf_preds))

# Support Vector Machine Model
svm = SVC(kernel='rbf', class_weight='balanced')
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)

print("\n🎯 SVM Results:")
print(confusion_matrix(y_test, svm_preds))
print(classification_report(y_test, svm_preds))
print("Accuracy:", accuracy_score(y_test, svm_preds))
