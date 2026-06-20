import pandas as pd

# 1. Load Dataset
df = pd.read_csv(
    r"D:\SEM_5\ML\datasets\Data cleaning\employee\WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

print("Original Shape:", df.shape)

records_before = len(df)

# 2. Missing Values
print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())

missing_before = df.isnull().sum().sum()

for col in df.columns:

    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])

    else:
        df[col] = df[col].fillna(df[col].median())

missing_after = df.isnull().sum().sum()

# 3. Remove Duplicate Records
duplicate_count = df.duplicated().sum()

df = df.drop_duplicates()

print("\nDuplicates Removed:", duplicate_count)

# 4. Standardize Categorical Values
if "Gender" in df.columns:
    df["Gender"] = (
        df["Gender"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "male": "Male",
            "female": "Female"
        })
    )

if "Department" in df.columns:
    df["Department"] = (
        df["Department"]
        .astype(str)
        .str.strip()
        .str.title()
    )

# 5. Check and Correct Data Types
print("\n========== DATA TYPES BEFORE ==========")
df.info()

for col in df.columns:

    if (
        "Age" in col
        or "Income" in col
        or "Rate" in col
        or "Years" in col
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n========== DATA TYPES AFTER ==========")
df.info()

# 6. Detect and Remove Salary Outliers Using IQR Method
salary_col = "MonthlyIncome"

Q1 = df[salary_col].quantile(0.25)
Q3 = df[salary_col].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

outliers_removed = len(
    df[
        (df[salary_col] < lower_bound)
        | (df[salary_col] > upper_bound)
    ]
)

df = df[
    (df[salary_col] >= lower_bound)
    & (df[salary_col] <= upper_bound)
]

# 7. Save Cleaned Dataset
df.to_csv("cleaned_employee.csv", index=False)

records_after = len(df)

# 8. Cleaning Summary
print("\n========== CLEANING SUMMARY ==========")

print("Records Before Cleaning :", records_before)
print("Records After Cleaning  :", records_after)

print("Missing Values Found    :", missing_before)
print("Missing Values Removed  :", missing_before - missing_after)

print("Duplicates Removed      :", duplicate_count)

print("Salary Outliers Removed :", outliers_removed)

print("\nCleaned dataset saved as:")
print("cleaned_employee.csv")