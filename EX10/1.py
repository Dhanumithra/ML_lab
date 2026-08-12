# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = sns.load_dataset('titanic')

print("First 5 rows")
print(df.head())

print("\nDataset Shape:", df.shape)

# Select useful features
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

# Handle Missing Values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Convert Categorical Columns
encoder = LabelEncoder()
df['sex'] = encoder.fit_transform(df['sex'])
df['embarked'] = encoder.fit_transform(df['embarked'])

# Split Features and Target
X = df.drop('survived', axis=1)
y = df['survived']

# Define K-Fold Cross Validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Lists to store results per fold
fold_accuracies = []
y_true_all = []
y_pred_all = []

# Train Logistic Regression Model using K-Fold Loop
model = LogisticRegression(max_iter=1000)

print("\n--- Fold-wise Accuracy Results ---")
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
    # Split data for current fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train Model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy for the current fold
    acc = accuracy_score(y_test, y_pred) * 100
    fold_accuracies.append(acc)
    
    # Store labels for overall matrix & report
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
    
    print(f"Fold {fold} Accuracy : {round(acc, 2)} %")

# Overall Accuracy
mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print("\nAverage K-Fold Accuracy :", round(mean_acc, 2), "%")
print("Standard Deviation      :", round(std_acc, 2), "%")

# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(y_true_all, y_pred_all))

# Classification Report
print("\nClassification Report")
print(classification_report(y_true_all, y_pred_all))

# Accuracy Graph across Folds
folds = [f"Fold {i}" for i in range(1, k + 1)]

plt.figure(figsize=(7, 5))
plt.bar(folds, fold_accuracies, color='skyblue', edgecolor='black')

# Draw line for mean accuracy
plt.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean Acc: {mean_acc:.2f}%')

plt.ylabel("Accuracy (%)")
plt.title("K-Fold Cross Validation Accuracy per Fold")
plt.ylim(0, 100)
plt.legend(loc='lower right')

for i, v in enumerate(fold_accuracies):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=10)

plt.show()