# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Load Dataset
# -------------------------------
df = sns.load_dataset('titanic')

print("First 5 rows")
print(df.head())

print("\nDataset Shape:", df.shape)

# -------------------------------
# Select useful features
# -------------------------------
df = df[['survived','pclass','sex','age','sibsp','parch','fare','embarked']]

# -------------------------------
# Handle Missing Values
# -------------------------------
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# -------------------------------
# Convert Categorical Columns
# -------------------------------
encoder = LabelEncoder()

df['sex'] = encoder.fit_transform(df['sex'])
df['embarked'] = encoder.fit_transform(df['embarked'])

# -------------------------------
# Split Features and Target
# -------------------------------
X = df.drop('survived', axis=1)
y = df['survived']

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Train Logistic Regression Model
# -------------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Accuracy
# -------------------------------
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("\nTraining Accuracy :", round(train_acc*100,2),"%")
print("Testing Accuracy :", round(test_acc*100,2),"%")

# -------------------------------
# Confusion Matrix
# -------------------------------
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Classification Report
# -------------------------------
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# -------------------------------
# Accuracy Graph
# -------------------------------
accuracy = [train_acc*100, test_acc*100]
labels = ['Training Accuracy','Testing Accuracy']

plt.figure(figsize=(6,5))
plt.bar(labels, accuracy)

plt.ylabel("Accuracy (%)")
plt.title("Training vs Testing Accuracy")
plt.ylim(0,100)

for i,v in enumerate(accuracy):
    plt.text(i, v+1, f"{v:.2f}%", ha='center', fontsize=11)

plt.show()