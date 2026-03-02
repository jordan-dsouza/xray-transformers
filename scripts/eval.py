import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification

from xray.config import TrainingConfig
from xray.data import load_imdb


def evaluate(model_path: str):
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader, _ = load_imdb(
        tokenizer_name=config.model_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probs = torch.softmax(outputs.logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Avg Confidence: {np.mean(all_confidences):.4f}")


if __name__ == "__main__":
    evaluate("models/distilbert_epoch_3.pt")