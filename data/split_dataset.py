import pandas as pd

INPUT_PATH = "data/training_clean.csv"
TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"

TEST_SIZE = 0.15      # 15% validation
RANDOM_STATE = 42

df = pd.read_csv(INPUT_PATH)

labels = df["label"].unique()
train_parts = []
val_parts = []

for label in labels:
    group = df[df["label"] == label]
    # sample some rows for validation
    val_group = group.sample(
        frac=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_group = group.drop(val_group.index)

    train_parts.append(train_group)
    val_parts.append(val_group)

train_df = pd.concat(train_parts).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
val_df = pd.concat(val_parts).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("\nTrain label distribution:")
print(train_df["label"].value_counts())
print("\nVal label distribution:")
print(val_df["label"].value_counts())