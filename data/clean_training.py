import pandas as pd

df = pd.read_csv("data/training.csv")

def clean(t):
    if isinstance(t, str):
        return t.replace("\\n", " ").replace("\n", " ").strip()
    return t

df["text"] = df["text"].apply(clean)

df.to_csv("data/training_clean.csv", index=False)
print("Saved cleaned dataset: training_clean.csv")