import torch
import numpy as np


def predict_target_logit(
    model,
    tokenizer,
    text,
    target_label,
    device
):
    """
    Return the model's target-class logit for a text input.
    """

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    return outputs.logits[0, target_label].item()


def _get_word_token_groups(tokens):
    """
    Group original WordPiece tokens into words.

    Returns:
        words:
            Reconstructed word strings.

        word_token_groups:
            Original token positions belonging to each word.
    """

    words = []
    word_token_groups = []

    current_word = None
    current_indices = []

    for index, token in enumerate(tokens):

        # Ignore special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # Start a new word
        if not token.startswith("##"):

            if current_word is not None:
                words.append(current_word)
                word_token_groups.append(current_indices)

            current_word = token
            current_indices = [index]

        # Continue current WordPiece word
        else:

            if current_word is not None:
                current_word += token[2:]
                current_indices.append(index)

    # Store final word
    if current_word is not None:
        words.append(current_word)
        word_token_groups.append(current_indices)

    return words, word_token_groups


def _create_deleted_input(
    input_ids,
    attention_mask,
    deleted_token_indices,
    tokenizer
):
    """
    Create a new model input after removing selected tokens.

    Special tokens are preserved.
    """

    deleted_token_indices = set(deleted_token_indices)

    original_ids = input_ids[0].tolist()

    remaining_ids = [
        token_id
        for index, token_id in enumerate(original_ids)
        if index not in deleted_token_indices
    ]

    new_input_ids = torch.tensor(
        [remaining_ids],
        dtype=input_ids.dtype,
        device=input_ids.device
    )

    new_attention_mask = torch.ones_like(
        new_input_ids
    )

    return new_input_ids, new_attention_mask


def _predict_from_ids(
    model,
    input_ids,
    attention_mask,
    target_label
):
    """
    Predict target-class logit directly from token IDs.
    """

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    return outputs.logits[0, target_label].item()


def top_k_deletion_curve(
    model,
    tokenizer,
    text,
    tokens,
    scores,
    target_label,
    device
):
    """
    Measure target-class logit while deleting words
    in descending attribution magnitude.

    The original tokenization is preserved.
    """

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    words, word_token_groups = _get_word_token_groups(tokens)

    scores = np.asarray(
        [float(score) for score in scores],
        dtype=float
    )

    if len(words) != len(scores):
        raise ValueError(
            "Number of words and attribution scores must match."
        )

    ranking = np.argsort(-np.abs(scores))

    original_logit = _predict_from_ids(
        model,
        input_ids,
        attention_mask,
        target_label
    )

    logits = [original_logit]
    fractions = [0.0]

    deleted_token_indices = []

    for step, word_index in enumerate(ranking, start=1):

        deleted_token_indices.extend(
            word_token_groups[int(word_index)]
        )

        new_input_ids, new_attention_mask = _create_deleted_input(
            input_ids,
            attention_mask,
            deleted_token_indices,
            tokenizer
        )

        logit = _predict_from_ids(
            model,
            new_input_ids,
            new_attention_mask,
            target_label
        )

        logits.append(logit)
        fractions.append(step / len(words))

    return {
        "words": words,
        "scores": scores,
        "fractions": fractions,
        "logits": logits,
        "original_logit": original_logit
    }


def random_deletion_curve(
    model,
    tokenizer,
    text,
    tokens,
    target_label,
    device,
    seed=42
):
    """
    Measure target-class logit while randomly deleting words.

    The original tokenization is preserved.
    """

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    words, word_token_groups = _get_word_token_groups(tokens)

    rng = np.random.default_rng(seed)

    ranking = np.arange(len(words))
    rng.shuffle(ranking)

    original_logit = _predict_from_ids(
        model,
        input_ids,
        attention_mask,
        target_label
    )

    logits = [original_logit]
    fractions = [0.0]

    deleted_token_indices = []

    for step, word_index in enumerate(ranking, start=1):

        deleted_token_indices.extend(
            word_token_groups[int(word_index)]
        )

        new_input_ids, new_attention_mask = _create_deleted_input(
            input_ids,
            attention_mask,
            deleted_token_indices,
            tokenizer
        )

        logit = _predict_from_ids(
            model,
            new_input_ids,
            new_attention_mask,
            target_label
        )

        logits.append(logit)
        fractions.append(step / len(words))

    return {
        "words": words,
        "fractions": fractions,
        "logits": logits,
        "original_logit": original_logit
    }

def repeated_random_deletion_curves(
    model,
    tokenizer,
    text,
    tokens,
    target_label,
    device,
    seeds
):
    """
    Run random deletion multiple times using different seeds.

    Returns a list of random deletion curves.
    """

    curves = []

    for seed in seeds:

        curve = random_deletion_curve(
            model=model,
            tokenizer=tokenizer,
            text=text,
            tokens=tokens,
            target_label=target_label,
            device=device,
            seed=seed
        )

        curves.append(curve)

    return curves

def summarize_random_curves(random_curves):
    """
    Calculate the mean and standard deviation across
    multiple random deletion curves.
    """

    logits = np.asarray([
        curve["logits"]
        for curve in random_curves
    ])

    fractions = np.asarray(
        random_curves[0]["fractions"],
        dtype=float
    )

    mean_logits = np.mean(
        logits,
        axis=0
    )

    std_logits = np.std(
        logits,
        axis=0
    )

    return {
        "fractions": fractions,
        "mean_logits": mean_logits,
        "std_logits": std_logits,
        "all_logits": logits
    }

def evaluate_logit_faithfulness(
    top_k_curve,
    random_curve
):
    """
    Compare attribution-guided deletion against random deletion.

    Measures the area under the cumulative target-logit drop curve.

    A larger AUC means the target logit drops more rapidly
    as important features are deleted.
    """

    top_k_logits = np.asarray(
        top_k_curve["logits"],
        dtype=float
    )

    random_logits = np.asarray(
        random_curve["logits"],
        dtype=float
    )

    fractions = np.asarray(
        top_k_curve["fractions"],
        dtype=float
    )

    original_logit = float(
        top_k_curve["original_logit"]
    )

    # Convert logits into cumulative drops.
    top_k_drops = (
        original_logit - top_k_logits
    )

    random_drops = (
        original_logit - random_logits
    )

    # Area under cumulative-drop curves.
    top_k_auc = np.trapezoid(
        top_k_drops,
        fractions
    )

    random_auc = np.trapezoid(
        random_drops,
        fractions
    )

    return {
        "original_logit": original_logit,

        "top_k_final_logit": top_k_logits[-1],
        "random_final_logit": random_logits[-1],

        "top_k_final_drop": top_k_drops[-1],
        "random_final_drop": random_drops[-1],

        "top_k_drop_auc": top_k_auc,
        "random_drop_auc": random_auc,

        "top_k_auc_higher": (
            top_k_auc > random_auc
        )
    }