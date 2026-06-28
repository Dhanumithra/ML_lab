import seaborn as sns

print("Before")
df = sns.load_dataset("titanic")
print("After")
print(df.head())