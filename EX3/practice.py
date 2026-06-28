import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Load Titanic Dataset

df = sns.load_dataset('titanic')

df = df[['age', 'fare', 'sex', 'class', 'embark_town']]

df = df.dropna()

print("========== ORIGINAL DATASET ==========")
print(df.head())

df_processed = df.copy()

# Min-Max Scaling

minmax = MinMaxScaler()
df_processed[['age']] = minmax.fit_transform(df_processed[['age']])

# Standard Scaling

standard = StandardScaler()
df_processed[['fare']] = standard.fit_transform(df_processed[['fare']])

# Label Encoding

label_encoder = LabelEncoder()
df_processed['sex'] = label_encoder.fit_transform(df_processed['sex'])

# One-Hot Encoding

df_processed = pd.get_dummies(
    df_processed,
    columns=['embark_town', 'class']
)

print("\n========== TRANSFORMED DATASET ==========")
print(df_processed.head())

print("\nOriginal Shape :", df.shape)
print("Transformed Shape :", df_processed.shape)