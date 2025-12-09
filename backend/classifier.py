# backend/classifier.py

import os
import torch
import torch.nn.functional as F
from typing import Optional
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


class EmailClassifier:
    def __init__(self, model_dir: Optional[str] = None):
        """
        model_dir: path to the folder containing the fine-tuned DistilBERT model
        and tokenizer (i.e., where save_pretrained() stored the files).
        """
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(__file__),
                "..",
                "models",
                "bert_model",
            )

        model_dir = os.path.abspath(model_dir)
        print("Loading model from:", model_dir)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)

        # Device
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print("Using device:", self.device)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        """
        Run the classifier on a single email.
        Returns dict: label, confidence, probabilities.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]

        confidence, label_id = torch.max(probs, dim=-1)
        label = self.model.config.id2label[label_id.item()]

        return {
            "label": label,
            "confidence": round(confidence.item(), 4),
            "probabilities": probs.cpu().numpy().tolist(),
        }


if __name__ == "__main__":
    clf = EmailClassifier()
    print(clf.predict("I was billed twice. Please check."))