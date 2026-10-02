from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    dataset_name: str = "imdb"
    max_length: int = 256
    batch_size: int = 32
    lr: float = 2e-5
    epochs: int = 2
    seed: int = 42