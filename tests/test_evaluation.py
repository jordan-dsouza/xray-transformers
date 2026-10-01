import numpy as np
import torch

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))

from xray.evaluation import (
    _get_word_token_groups,
    _create_deleted_input,
    logit_to_probability,
    summarize_random_curves,
    evaluate_repeated_random_faithfulness,
    evaluate_probability_faithfulness,
)

# Test WordPiece Grouping:
def test_get_word_token_groups():
    """
    This verifies that deletion evaluation 
    correctly understands WordPiece tokens 
    while retaining the original token positions
    """
    tokens = [
        "[CLS]",
        "mar",
        "##vel",
        "##lous",
        "movie",
        "was",
        "great",
        "[SEP]"
    ]

    words, groups = _get_word_token_groups(tokens)

    assert words == [
        "marvellous",
        "movie",
        "was",
        "great"
    ]

    assert groups == [
        [1, 2, 3],
        [4],
        [5],
        [6]
    ]

# Test Probability Conversion:
def test_logit_to_probability():

    probability = logit_to_probability(0.0)

    assert np.isclose(
        probability,
        0.5
    )

def test_logit_to_probability_positive():

    probability = logit_to_probability(2.0)

    assert probability > 0.5
    assert probability < 1.0

# Test random curve summarization:

def test_summarize_random_curves():
    """
    This verifies that our repeated-random baseline is being summarized correctly.
    """
    random_curves = [
        {
            "fractions": [0.0, 0.5, 1.0],
            "logits": [1.0, 0.5, 0.0]
        },
        {
            "fractions": [0.0, 0.5, 1.0],
            "logits": [1.0, 0.3, -0.2]
        }
    ]

    summary = summarize_random_curves(
        random_curves
    )

    expected_mean = np.array([
        1.0,
        0.4,
        -0.1
    ])

    assert np.allclose(
        summary["mean_logits"],
        expected_mean
    )

    assert summary["all_logits"].shape == (2, 3)

# Test the faithfulness comparison:

def test_repeated_random_faithfulness():
    """
    This checks the central behavior we're interested in:
    attribution-guided deletion
        ↓
    larger cumulative target-logit drop
        ↓
    larger AUC

    """
    top_k_curve = {
        "fractions": [0.0, 0.5, 1.0],
        "logits": [2.0, 1.0, 0.0],
        "original_logit": 2.0
    }

    random_summary = {
        "fractions": np.array([
            0.0,
            0.5,
            1.0
        ]),
        "mean_logits": np.array([
            2.0,
            1.5,
            0.0
        ])
    }

    result = evaluate_repeated_random_faithfulness(
        top_k_curve,
        random_summary
    )

    assert result["top_k_drop_auc"] > (
        result["random_mean_drop_auc"]
    )

    assert result["top_k_auc_higher"] is True

# Test probability-based faithfulness:

def test_probability_faithfulness():

    top_k_curve = {
        "fractions": [0.0, 0.5, 1.0],
        "logits": [2.0, 1.0, 0.0],
        "original_logit": 2.0
    }

    random_summary = {
        "fractions": np.array([
            0.0,
            0.5,
            1.0
        ]),
        "mean_logits": np.array([
            2.0,
            1.5,
            0.0
        ])
    }

    result = evaluate_probability_faithfulness(
        top_k_curve,
        random_summary
    )

    assert (
        result["top_k_probability_drop_auc"]
        >
        result["random_probability_drop_auc"]
    )

    assert result["top_k_auc_higher"] is True