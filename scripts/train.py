import torch
import os
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm

from xray.config import TrainingConfig
from xray.utils import set_seed
from xray.data import load_imdb
from torch.amp import autocast, GradScaler

def train():
    os.makedirs("models", exist_ok=True)
    config = TrainingConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader, _ = load_imdb(
        tokenizer_name=config.model_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr)
    scaler = GradScaler('cuda')

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in loop:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            with autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg loss: {avg_loss:.4f}")

        # Save checkpoint every epoch
        torch.save(model.state_dict(), f"models/distilbert_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
