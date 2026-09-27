import torch

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer
)

from xray.interpretability import DistilBertInterpreter

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Loading tokenizer...")


# Load YOUR fine-tuned model HERE in "quotes" from HuggingFace:
model = DistilBertForSequenceClassification.from_pretrained(
    "JDsouza1/distilbert-imdb-sentiment"
).to(device)

print("loading model...")

# Load tokenizer used for model:
tokenizer = DistilBertTokenizer.from_pretrained(
    "JDsouza1/distilbert-imdb-sentiment"
)

print("Model loaded!")

# INTERPRETABILITY TOOL:
interpreter = DistilBertInterpreter(
    model, 
    tokenizer,
    device
)

# Eg text to explain:
text = "This movie was marvellous!"

# Calculate token importance scores:
tokens, attributions, delta = interpreter.attribute(text)

print("\nRaw token attributions:")
print("-" * 40)

# Print each token and attribution score:
for token, score in zip(tokens, attributions):
    print(f"{token:15} {float(score):.4f}")

# Merge WordPiece tokens:
words, scores = interpreter.aggregate_tokens(
    tokens,
    attributions
)

print("\nWord - level attributions:")
print("-" * 40)

for word, scores in zip(words, scores):
    print(f"{word:15} {float(score):.4f}")

# Check how well have the Integrated Gradients converged:
print("\nConvergence delta:", float(delta.item()))

# Delta value closer to 0 indicates better convergence