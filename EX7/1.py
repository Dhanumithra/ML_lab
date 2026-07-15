import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

np.random.seed(42)

n = 1000

study = np.random.randint(1, 11, n)
att = np.random.randint(50, 101, n)
assign = np.random.randint(0, 21, n)
internal = np.random.randint(0, 51, n)

score = (
    study * 3 +
    att * 0.3 +
    assign * 2 +
    internal * 1.5
)

result = np.where(score >= 90, "Pass", "Fail")

df = pd.DataFrame({
    "study": study,
    "att": att,
    "assign": assign,
    "internal": internal,
    "result": result
})

print("Dataset Sample")
print(df.head())

df["result"] = df["result"].map({"Fail": 0, "Pass": 1})

X = df.drop("result", axis=1)
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

full_tree = DecisionTreeClassifier(random_state=42)

full_tree.fit(X_train, y_train)

pred_full = full_tree.predict(X_test)

print("\nUNRESTRICTED DECISION TREE")
print("Accuracy :", accuracy_score(y_test, pred_full))
print("Precision:", precision_score(y_test, pred_full))
print("Recall   :", recall_score(y_test, pred_full))
print("F1 Score :", f1_score(y_test, pred_full))

cm = confusion_matrix(y_test, pred_full)

tn, fp, fn, tp = cm.ravel()

spec = tn / (tn + fp)

print("Specificity:", spec)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, pred_full))

pruned_tree = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

pruned_tree.fit(X_train, y_train)

pred_pruned = pruned_tree.predict(X_test)

print("\nPRE-PRUNED DECISION TREE (max_depth=4)")
print("Accuracy :", accuracy_score(y_test, pred_pruned))
print("Precision:", precision_score(y_test, pred_pruned))
print("Recall   :", recall_score(y_test, pred_pruned))
print("F1 Score :", f1_score(y_test, pred_pruned))

cm = confusion_matrix(y_test, pred_pruned)

tn, fp, fn, tp = cm.ravel()

spec = tn / (tn + fp)

print("Specificity:", spec)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, pred_pruned))