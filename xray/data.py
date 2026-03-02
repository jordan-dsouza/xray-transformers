from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def load_imdb(tokenizer_name: str, max_length: int = 256, batch_size: int = 16):
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    train_loader = DataLoader(
        tokenized["train"], batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        tokenized["test"], batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, tokenizer