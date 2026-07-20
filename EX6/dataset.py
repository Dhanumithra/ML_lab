import pandas as pd
import numpy as np

# For reproducible results
np.random.seed(42)

# Generate Years of Experience (0.5 to 20 years)
years_experience = np.round(np.linspace(0.5, 20, 50), 1)

# Generate Salary
# Formula: Salary = 30000 + (9500 * YearsExperience) + Random Noise
salary = 30000 + (9500 * years_experience) + np.random.randint(-5000, 5000, size=50)

# Create DataFrame
df = pd.DataFrame({
    "YearsExperience": years_experience,
    "Salary": salary
})

# Save to CSV
df.to_csv("Salary_Data.csv", index=False)

print("Dataset Generated Successfully!\n")
print(df.head())