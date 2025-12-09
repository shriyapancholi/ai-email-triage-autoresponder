import os
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# -------------------------------------------------------------------
# 0. Stub sklearn so that transformers doesn't try to import real one
#    (real scikit-learn + scipy are broken with your numpy version)
# -------------------------------------------------------------------
if "sklearn" not in sys.modules:
    sklearn_stub = types.ModuleType("sklearn")
    sklearn_stub.__spec__ = ModuleSpec("sklearn", loader=None)  # IMPORTANT

    metrics_stub = types.ModuleType("sklearn.metrics")

    def roc_curve(*args, **kwargs):
        # We don't actually need this; it's only used for generation helpers.
        raise NotImplementedError("roc_curve is not available in this environment.")

    metrics_stub.roc_curve = roc_curve

    # attach submodule
    sklearn_stub.metrics = metrics_stub

    # register both in sys.modules
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.metrics"] = metrics_stub

# -------------------------------------------------------------------
# 1. Now it's safe to import transformers
# -------------------------------------------------------------------
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# -----------------------
# Paths
# -----------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models" / "bert_model"
TOKENIZER_DIR = ROOT_DIR / "models" / "tokenizer"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Label mapping (FIXED)
# -----------------------
LABEL2ID = {
    "billing": 0,
    "technical_issue": 1,
    "product_info": 2,
    "other": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


# -----------------------
# Load data
# -----------------------
def load_data(path: Path):
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()
    df["label_id"] = df["label"].map(LABEL2ID)
    return df


train_df = load_data(TRAIN_PATH)
val_df = load_data(VAL_PATH)

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("\nTrain label distribution:\n", train_df["label"].value_counts())
print("\nVal label distribution:\n", val_df["label"].value_counts())

# -----------------------
# Tokenizer and model
# -----------------------
MODEL_NAME = "distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)


# -----------------------
# Dataset class
# -----------------------
class EmailDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


train_dataset = EmailDataset(train_df, tokenizer)
val_dataset = EmailDataset(val_df, tokenizer)

# -----------------------
# Class weights for imbalance
# -----------------------
label_counts = train_df["label_id"].value_counts().sort_index()
total = label_counts.sum()
class_weights = total / (len(label_counts) * label_counts)
class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float)

print("\nClass weights:", class_weights_tensor)


# Custom Trainer to use weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# -----------------------
# Training arguments
# -----------------------
training_args = TrainingArguments(
    output_dir=str(ROOT_DIR / "outputs"),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
)

# -----------------------
# Metrics WITHOUT sklearn
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Accuracy
    acc = (preds == labels).mean()

    # Macro F1 (manual)
    f1_scores = []
    for cls in range(NUM_LABELS):
        tp = np.sum((preds == cls) & (labels == cls))
        fp = np.sum((preds == cls) & (labels != cls))
        fn = np.sum((preds != cls) & (labels == cls))

        if tp == 0 and (fp > 0 or fn > 0):
            f1 = 0.0
        elif tp == 0 and fp == 0 and fn == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    f1_macro = float(np.mean(f1_scores))

    return {
        "accuracy": float(acc),
        "f1_macro": f1_macro,
    }


# -----------------------
# Trainer
# -----------------------
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# -----------------------
# Train
# -----------------------
if __name__ == "__main__":
    print("\nStarting training...")
    trainer.train()

    print("\nSaving model to:", MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

    print("Saving tokenizer to:", TOKENIZER_DIR)
    tokenizer.save_pretrained(TOKENIZER_DIR)

    print("\nDone.")