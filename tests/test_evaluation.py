import numpy as np
import torch

from xray.evaluation import (
    _get_word_token_groups,
    _create_deleted_input,
    logit_to_probability,
    summarize_random_curves,
    evaluate_repeated_random_faithfulness,
    evaluate_probability_faithfulness,
)