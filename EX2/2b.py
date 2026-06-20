import pandas as pd

# 1. Load Dataset
df = pd.read_csv(
    r"D:\SEM_5\ML\datasets\Data cleaning\Dirty Cars\archive\dirty_cafe_sales.csv"
)

print("Original Shape:", df.shape)

records_before = len(df)

# 2. Basic Information
print("\nDataset Info")
df.info()

# 3. Missing Values
print("\nMissing Values")
print(df.isnull().sum())

missing_before = df.isnull().sum().sum()

for col in df.columns:

    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])

missing_after = df.isnull().sum().sum()

missing_removed = missing_before - missing_after

# 4. Remove Duplicate Transactions
duplicates_removed = df.duplicated().sum()

df = df.drop_duplicates()

# 5. Convert Numeric Columns

df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

df["Price Per Unit"] = pd.to_numeric(
    df["Price Per Unit"],
    errors="coerce"
)

df["Total Spent"] = pd.to_numeric(
    df["Total Spent"],
    errors="coerce"
)

# Fill numeric missing values
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())

df["Price Per Unit"] = df["Price Per Unit"].fillna(
    df["Price Per Unit"].median()
)

df["Total Spent"] = df["Total Spent"].fillna(
    df["Total Spent"].median()
)

# 6. Remove Invalid Transactions

invalid_transactions = len(
    df[
        (df["Quantity"] <= 0)
        | (df["Price Per Unit"] <= 0)
    ]
)

df = df[
    (df["Quantity"] > 0)
    & (df["Price Per Unit"] > 0)
]

# 7. Convert Transaction Date

df["Transaction Date"] = pd.to_datetime(
    df["Transaction Date"],
    errors="coerce"
)

invalid_dates = df["Transaction Date"].isnull().sum()

df = df.dropna(subset=["Transaction Date"])

# 8. Standardize Item Names

df["Item"] = (
    df["Item"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Standardize Payment Method

df["Payment Method"] = (
    df["Payment Method"]
    .astype(str)
    .str.strip()
    .str.title()
)

# Standardize Location

df["Location"] = (
    df["Location"]
    .astype(str)
    .str.strip()
    .str.title()
)

# 9. Save Clean Dataset

df.to_csv("cleaned_sales_dataset.csv", index=False)

records_after = len(df)

# 10. Cleaning Summary

print("\n========== CLEANING SUMMARY ==========")

print("Records Before Cleaning :", records_before)
print("Records After Cleaning  :", records_after)

print("Missing Values Removed  :", missing_removed)

print("Duplicate Records Removed :", duplicates_removed)

print("Invalid Transactions Removed :", invalid_transactions)

print("Invalid Dates Removed :", invalid_dates)

print("\nCleaned dataset saved as:")
print("cleaned_sales_dataset.csv")