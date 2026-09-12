import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the built-in scikit-learn diabetes dataset
diabetes = load_diabetes(as_frame=True)
X = diabetes.data

# Convert continuous progression target to binary classes (0 = Low Risk, 1 = High Risk)
y = (diabetes.target > np.median(diabetes.target)).astype(int)

# 2. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Naive Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Display evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))