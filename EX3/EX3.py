import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Load Dataset

df = pd.read_excel(r"D:\SEM_5\ML\datasets\Data Preperation\emp-data.xlsx")

print("========== ORIGINAL DATASET ==========")
print(df.head())

# Task 1: Identify Categorical and Numerical Attributes

print("\n========== TASK 1 ==========")

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

print("\nNumerical Attributes:")
print(list(numerical_columns))

print("\nCategorical Attributes:")
print(list(categorical_columns))

# Task 2: Label Encoding

print("\n========== TASK 2 ==========")

label_encoder = LabelEncoder()

df_label = df.copy()

df_label["Gender"] = label_encoder.fit_transform(df_label["Gender"])
df_label["Education"] = label_encoder.fit_transform(df_label["Education"])

print(df_label[["Gender", "Education"]].head())

# Task 3: One-Hot Encoding

print("\n========== TASK 3 ==========")

df_onehot = pd.get_dummies(
    df_label,
    columns=["Department", "City", "Work_Mode", "Job_Role"]
)

print(df_onehot.head())

# Task 4: StandardScaler

print("\n========== TASK 4 ==========")

standard_df = df_onehot.copy()

standard_scaler = StandardScaler()

columns_to_scale = [
    "Age",
    "Experience",
    "Salary",
    "Performance_Score"
]

standard_df[columns_to_scale] = standard_scaler.fit_transform(
    standard_df[columns_to_scale]
)

print(standard_df[columns_to_scale].head())

# Task 5: MinMaxScaler

print("\n========== TASK 5 ==========")

minmax_df = df_onehot.copy()

minmax_scaler = MinMaxScaler()

minmax_df[columns_to_scale] = minmax_scaler.fit_transform(
    minmax_df[columns_to_scale]
)

print(minmax_df[columns_to_scale].head())

# Task 6: Visualization

print("\n========== TASK 6 ==========")

plt.figure(figsize=(10,4))
plt.hist(df["Salary"], bins=10)
plt.title("Salary Before Standard Scaling")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10,4))
plt.hist(standard_df["Salary"], bins=10)
plt.title("Salary After Standard Scaling")
plt.xlabel("Scaled Salary")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.boxplot(df["Salary"])
plt.title("Salary Before Standard Scaling")
plt.show()

plt.figure(figsize=(6,4))
plt.boxplot(standard_df["Salary"])
plt.title("Salary After Standard Scaling")
plt.show()

print("\nInterpretation:")
print("StandardScaler changes the scale of Salary.")
print("The shape of the distribution remains the same.")
print("The mean becomes approximately 0 and the standard deviation becomes 1.")

# Task 7: Compare Label Encoding and One-Hot Encoding

print("\n========== TASK 7 ==========")

department_label = df.copy()

department_label["Department"] = LabelEncoder().fit_transform(
    department_label["Department"]
)

print("\nLabel Encoding:")
print(department_label[["Department"]].head())

department_onehot = pd.get_dummies(
    df[["Department"]],
    columns=["Department"]
)

print("\nOne-Hot Encoding:")
print(department_onehot.head())