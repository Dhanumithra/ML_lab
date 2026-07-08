import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')

# EDA
print(df.head())
print("\nRows and Columns:", df.shape)
print("\nTarget Column: survived")
print(df.info())
print(df.isnull().sum())

# Select required columns
df = df[['sex', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']]

# Data preprocessing
df = df.dropna()

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['embarked'] = le.fit_transform(df['embarked'])

# Features and Target
X = df.drop('survived', axis=1)
y = df['survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Display Tree
plt.figure(figsize=(18,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Died', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.show()

# Predict for new data
new_data = pd.DataFrame({
    'sex': [1],
    'pclass': [3],
    'age': [25],
    'sibsp': [0],
    'parch': [0],
    'fare': [15.5],
    'embarked': [2]
})

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("\nPrediction: Survived")
else:
    print("\nPrediction: Died")