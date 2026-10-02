from typing import Any, TypedDict

import torch
import numpy as np

# Functions intended for users of xray.evaluation:
__all__ = [
    "predict_target_logit",
    "top_k_deletion_curve",
    "random_deletion_curve",
    "repeated_random_deletion_curves",
    "summarize_random_curves",
    "evaluate_repeated_random_faithfulness",
    "evaluate_probability_faithfulness",
    "evaluate_example",
    "logit_to_probability",
    "print_faithfulness_summary",
]

class DeletionCurve(TypedDict):
    words: list[str]
    scores: np.ndarray
    fractions: np.ndarray
    logits: np.ndarray
    original_logit: float


class RandomCurveSummary(TypedDict):
    mean_logits: np.ndarray
    std_logits: np.ndarray
    all_logits: np.ndarray


class FaithfulnessSummary(TypedDict):
    original_logit: float
    final_top_k_logit: float
    final_random_logit: float
    top_k_final_drop: float
    random_final_drop: float
    top_k_auc: float
    random_auc: float
    top_k_auc_higher: bool


class ProbabilityFaithfulnessSummary(TypedDict):
    original_probability: float
    final_top_k_probability: float
    final_random_probability: float
    top_k_final_drop: float
    random_final_drop: float
    top_k_auc: float
    random_auc: float
    top_k_auc_higher: bool

class ExampleEvaluation(TypedDict):
    text: str
    target_label: int
    tokens: list[str]
    token_attributions: np.ndarray
    completeness_error: float
    top_k_curve: DeletionCurve
    random_curves: list[DeletionCurve]
    random_summary: RandomCurveSummary
    logit_faithfulness: FaithfulnessSummary
    probability_faithfulness: ProbabilityFaithfulnessSummary

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


def _get_word_token_groups(tokens: list[str]) -> list[list[int]]:
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
    input_ids: torch.Tensor,
    deleted_token_indices: list[int] | set[int],
) -> tuple[torch.Tensor, torch.Tensor]:
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
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    scores: np.ndarray,
    target_label: int,
    device: torch.device,
) -> DeletionCurve:
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
            deleted_token_indices
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
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    target_label: int,
    device: torch.device,
    seed: int = 42,
) -> DeletionCurve:
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
            deleted_token_indices
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
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    target_label: int,
    device: torch.device,
    seeds: list[int],
) -> list[DeletionCurve]:
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


def summarize_random_curves(
    random_curves: list[dict[str, Any]],
) -> RandomCurveSummary:
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

def evaluate_repeated_random_faithfulness(
    top_k_curve: dict[str, Any],
    random_summary: dict[str, Any],
) -> FaithfulnessSummary:
    """
    Compare attribution-guided deletion against
    the mean of multiple random deletion curves.

    Uses area under the cumulative target-logit
    drop curve.
    """

    top_k_logits = np.asarray(
        top_k_curve["logits"],
        dtype=float
    )

    random_mean_logits = np.asarray(
        random_summary["mean_logits"],
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

    random_mean_drops = (
        original_logit - random_mean_logits
    )

    # Calculate area under cumulative-drop curves.
    top_k_auc = np.trapezoid(
        top_k_drops,
        fractions
    )

    random_mean_auc = np.trapezoid(
        random_mean_drops,
        fractions
    )

    return {
        "original_logit": original_logit,

        "top_k_final_logit": top_k_logits[-1],
        "random_mean_final_logit": random_mean_logits[-1],

        "top_k_final_drop": top_k_drops[-1],
        "random_mean_final_drop": random_mean_drops[-1],

        "top_k_drop_auc": top_k_auc,
        "random_mean_drop_auc": random_mean_auc,

        "top_k_auc_higher": (
            top_k_auc > random_mean_auc
        )
    }


def logit_to_probability(logit):
    """Convert logits to probabilities using the sigmoid function."""
    return torch.sigmoid(torch.as_tensor(logit))

def evaluate_probability_faithfulness(
    top_k_curve: dict[str, Any],
    random_summary: dict[str, Any],
) -> ProbabilityFaithfulnessSummary:
    """
    Compare attribution-guided deletion against
    repeated random deletion using target-class probability.

    Measures the area under the cumulative probability-drop curve.
    """

    top_k_logits = np.asarray(
        top_k_curve["logits"],
        dtype=float
    )

    random_mean_logits = np.asarray(
        random_summary["mean_logits"],
        dtype=float
    )

    fractions = np.asarray(
        top_k_curve["fractions"],
        dtype=float
    )

    # Convert logits to probabilities.
    top_k_probabilities = logit_to_probability(
        top_k_logits
    )

    random_probabilities = logit_to_probability(
        random_mean_logits
    )

    original_probability = float(
        top_k_probabilities[0]
    )

    # Probability decrease relative to original input.
    top_k_drops = (
        original_probability -
        top_k_probabilities
    )

    random_drops = (
        original_probability -
        random_probabilities
    )

    # Area under cumulative probability-drop curves.
    top_k_auc = np.trapezoid(
        top_k_drops,
        fractions
    )

    random_auc = np.trapezoid(
        random_drops,
        fractions
    )

    return {
        "original_probability": original_probability,

        "top_k_final_probability": (
            top_k_probabilities[-1]
        ),

        "random_mean_final_probability": (
            random_probabilities[-1]
        ),

        "top_k_final_probability_drop": (
            top_k_drops[-1]
        ),

        "random_mean_final_probability_drop": (
            random_drops[-1]
        ),

        "top_k_probability_drop_auc": top_k_auc,

        "random_probability_drop_auc": random_auc,

        "top_k_auc_higher": (
            top_k_auc > random_auc
        )
    }


def print_faithfulness_summary(summary):
    """
    Print a compact summary of faithfulness evaluation.
    """

    print("\nFAITHFULNESS EVALUATION")
    print("-" * 40)

    print(f"Original target logit:       {summary['original_logit']:.4f}")
    print(
        f"Attribution final logit:     "
        f"{summary['top_k_final_logit']:.4f}"
    )
    print(
        f"Mean random final logit:     "
        f"{summary['random_mean_final_logit']:.4f}"
    )

    print()

    print(
        f"Attribution drop AUC:        "
        f"{summary['top_k_drop_auc']:.4f}"
    )
    print(
        f"Mean random drop AUC:        "
        f"{summary['random_mean_drop_auc']:.4f}"
    )

    print()

    print(
        f"Attribution AUC > random:   "
        f"{summary['top_k_auc_higher']}"
    )



def evaluate_example(
    model: torch.nn.Module,
    tokenizer: Any,
    interpreter: Any,
    text: str,
    device: torch.device,
    seeds: list[int],
) -> ExampleEvaluation:
    """
    Run the complete attribution faithfulness evaluation
    for a single text example.

    Steps:
        1. Generate Integrated Gradients attributions.
        2. Aggregate WordPiece tokens into words.
        3. Determine the target class.
        4. Run attribution-guided deletion.
        5. Run repeated random deletion.
        6. Compare attribution against the random baseline.
    """

    # --------------------------------------------------
    # 1. Generate attributions
    # --------------------------------------------------

    (
        tokens,
        token_attributions,
        delta,
        input_output,
        baseline_output,
        total_attribution,
        completeness_error
    ) = interpreter.attribute(text)

    # --------------------------------------------------
    # 2. Determine target label
    # --------------------------------------------------

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

    target_label = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    # --------------------------------------------------
    # 3. Aggregate token attributions into words
    # --------------------------------------------------

    words, word_scores = interpreter.aggregate_tokens(
        tokens,
        token_attributions
    )

    # --------------------------------------------------
    # 4. Attribution-guided deletion
    # --------------------------------------------------

    top_k_curve = top_k_deletion_curve(
        model=model,
        tokenizer=tokenizer,
        text=text,
        tokens=tokens,
        scores=word_scores,
        target_label=target_label,
        device=device
    )

    # --------------------------------------------------
    # 5. Repeated random deletion
    # --------------------------------------------------

    random_curves = repeated_random_deletion_curves(
        model=model,
        tokenizer=tokenizer,
        text=text,
        tokens=tokens,
        target_label=target_label,
        device=device,
        seeds=seeds
    )

    # --------------------------------------------------
    # 6. Summarize random curves
    # --------------------------------------------------

    random_summary = summarize_random_curves(
        random_curves
    )

    # --------------------------------------------------
    # 7. Calculate faithfulness
    # --------------------------------------------------

    faithfulness = evaluate_repeated_random_faithfulness(
        top_k_curve=top_k_curve,
        random_summary=random_summary
    )

    # --------------------------------------------------
    # 8. Calculate probability faithfulness
    # --------------------------------------------------

    probability_faithfulness = evaluate_probability_faithfulness(
    top_k_curve=top_k_curve,
    random_summary=random_summary
)
    # --------------------------------------------------
    # 9. Return everything
    # --------------------------------------------------

    return {
        "text": text,
        "target_label": target_label,

        "tokens": tokens,
        "token_attributions": token_attributions,

        "words": words,
        "word_scores": word_scores,

        "convergence_delta": delta,
        "input_output": input_output,
        "baseline_output": baseline_output,
        "total_attribution": total_attribution,
        "completeness_error": completeness_error,

        "top_k_curve": top_k_curve,

        "random_curves": random_curves,
        "random_summary": random_summary,

        "faithfulness": faithfulness,
        "probability_faithfulness": probability_faithfulness
    }


