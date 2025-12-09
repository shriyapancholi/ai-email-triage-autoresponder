import pandas as pd

df = pd.read_csv("data/training_clean.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nDataset size:", len(df))