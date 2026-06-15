# 1. Load the dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\SEM_5\ML\datasets\Table_data\student_data.csv")

# 2. Display the first 10 records
print("First 10 Records:")
print(df.head(10))

# 3. Number of rows and columns
rows, cols = df.shape
print("\nNumber of Rows:", rows)
print("Number of Columns:", cols)

# 4. List all attributes
print("\nAttributes (Columns):")
print(df.columns.tolist())

# 5. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Total missing values
print("\nTotal Missing Values:", df.isnull().sum().sum())

# 6. Basic Statistics for Marks Column (G3)
print("\nBasic Statistics for Marks (G3):")

print("Mean Marks:", df['G3'].mean())
print("Median Marks:", df['G3'].median())
print("Minimum Marks:", df['G3'].min())
print("Maximum Marks:", df['G3'].max())

# 7. Identify Numerical and Categorical Attributes
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("\nNumerical Attributes:")
print(list(numerical_cols))

print("\nCategorical Attributes:")
print(list(categorical_cols))

# 8. Draw Histogram of Marks
if 'G3' in df.columns:
    plt.figure(figsize=(8,5))
    plt.hist(df['G3'], bins=10)
    plt.title("Histogram of Final Grade (G3)")
    plt.xlabel("G3 Marks")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
else:
    print("\nColumn 'G3' not found. Use the appropriate marks column.")

# 9. Correlation Between Numerical Attributes
correlation_matrix = df[numerical_cols].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# 10. Possible Target Variables for Prediction
print("\nPossible Target Variables:")

possible_targets = []

for col in ['G1', 'G2', 'G3']:
    if col in df.columns:
        possible_targets.append(col)

print(possible_targets)