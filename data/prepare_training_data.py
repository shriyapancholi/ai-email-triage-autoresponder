import pandas as pd
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT_DIR / "data" / "raw" / "customer_support_tickets.csv"
OUT_PATH = ROOT_DIR / "data" / "training.csv"

print("Reading:", RAW_PATH)
df = pd.read_csv(RAW_PATH)

print("Original columns:", df.columns.tolist())
print("Sample rows:")
print(df.head())

# ---- 1. Keep only English tickets (if 'language' column exists) ----
if "language" in df.columns:
    df = df[df["language"] == "en"]

# ---- 2. Decide which text column to use ----
# Replace 'message' with the actual column name that contains the email text
TEXT_COL_CANDIDATES = ["message", "text", "body", "ticket_text", "content"]

text_col = None
for c in TEXT_COL_CANDIDATES:
    if c in df.columns:
        text_col = c
        break

if text_col is None:
    raise ValueError("Could not find a text column. Check the CSV columns and update TEXT_COL_CANDIDATES.")

# ---- 3. Map original category/department to our 6 labels ----
# Replace 'category' or 'department' with whatever the dataset uses
LABEL_SOURCE_CANDIDATES = ["category", "department", "queue", "type"]

label_src = None
for c in LABEL_SOURCE_CANDIDATES:
    if c in df.columns:
        label_src = c
        break

if label_src is None:
    raise ValueError("Could not find a label source column. Check the CSV columns and update LABEL_SOURCE_CANDIDATES.")

print("Using text column:", text_col)
print("Using label source column:", label_src)
print("Unique raw labels:", df[label_src].unique()[:20])

# Example mapping – you'll tweak this based on actual labels in your CSV
mapping = {
    "billing": "billing",
    "invoice": "billing",
    "payment": "billing",
    "technical": "technical_issue",
    "bug": "technical_issue",
    "error": "technical_issue",
    "account": "account_access",
    "login": "account_access",
    "sales": "product_info",
    "info": "product_info",
    "complaint": "complaint",
    "refund": "complaint",
}

def map_label(raw_label: str) -> str:
    if pd.isna(raw_label):
        return "other"
    raw = str(raw_label).lower()
    for key, target in mapping.items():
        if key in raw:
            return target
    return "other"

df["label"] = df[label_src].apply(map_label)

# ---- 4. Build final dataframe ----
clean = df[[text_col, "label"]].rename(columns={text_col: "text"}).dropna()
clean = clean[clean["text"].str.strip().astype(bool)]

# Add id
clean = clean.reset_index(drop=True)
clean.insert(0, "id", clean.index + 1)

print("Label distribution:")
print(clean["label"].value_counts())

# ---- 5. Save to training.csv ----
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
clean.to_csv(OUT_PATH, index=False)
print("Saved cleaned training data to:", OUT_PATH)