# X-Ray: Interpretability & Uncertainty for Transformers

X-Ray is an open-source research toolkit for analyzing transformer-based models with a focus on **interpretability, attribution analysis, robustness, and uncertainty**.

The project currently focuses on transformer-based text classification and uses DistilBERT as the primary experimental model.

## Project Goals

Transformer models can achieve strong predictive performance while making it difficult to understand which parts of an input influenced a prediction.

X-Ray explores methods for:

* Understanding model predictions through feature attribution
* Evaluating attribution faithfulness through input perturbation
* Investigating robustness and model behavior under changes to the input
* Exploring uncertainty estimation and model confidence
* Building reproducible research tooling for transformer analysis

The emphasis is on **interpretability and research methodology rather than state-of-the-art predictive performance**.

## Current Scope

The current implementation focuses on:

* Text classification
* DistilBERT
* Integrated Gradients
* Word-level attribution analysis
* Attribution visualization
* Deletion-based attribution faithfulness evaluation
* Repeated random deletion baselines
* Logit- and probability-based faithfulness metrics
* Robustness experiments
* Automated tests for evaluation utilities

## Project Structure

```text
xray-transformers/
├── notebooks/
│   ├── explain.ipynb
│   ├── exploration.ipynb
│   └── robustness.ipynb
│
├── scripts/
│   ├── eval.py
│   ├── interpret.py
│   └── train.py
│
├── tests/
│   ├── test_dummy.py
│   └── test_evaluation.py
│
├── xray/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── interpretability.py
│   └── utils.py
│
├── README.md
├── pyproject.toml
└── LICENSE
```

## Model and Dataset

The primary experiments use a **DistilBERT** model fine-tuned for binary sentiment classification on the **IMDb movie review dataset**.

The trained model is available through Hugging Face:

**JDsouza1/distilbert-imdb-sentiment**

The model is used for experimentation with attribution, perturbation, and robustness analysis.

## Interpretability

X-Ray currently implements **Integrated Gradients** for transformer text classification.

The interpretation pipeline:

```text
Input text
    ↓
DistilBERT tokenizer
    ↓
Input embeddings
    ↓
Integrated Gradients
    ↓
Token-level attributions
    ↓
Word-level aggregation
    ↓
Visualization / analysis
```

Integrated Gradients is computed with respect to the model's input embeddings.

The implementation also monitors the Integrated Gradients completeness property by comparing the sum of the attributions with the difference between the model output for the input and the chosen baseline.

### Word-level Attribution

DistilBERT uses WordPiece tokenization, so a single word can be represented by multiple subword tokens.

X-Ray aggregates these subword attributions to produce word-level explanations.

For example:

```text
marvellous → marvellous
```

or, when tokenized into multiple WordPieces:

```text
mar + ##vellous → marvellous
```

This makes the resulting explanations easier to inspect at the natural word level.

## Attribution Faithfulness Evaluation

The project evaluates whether token attributions identify inputs that are important to the model's prediction.

The evaluation uses a deletion-based approach:

1. Generate token-level attributions using Integrated Gradients.
2. Aggregate subword attributions into word-level scores.
3. Rank words by the absolute value of their attribution.
4. Iteratively delete words in attribution-ranked order.
5. Record the model's target-class logit after each deletion.
6. Repeat the deletion process using randomly ordered words as a baseline.
7. Run multiple random seeds to reduce dependence on a single random ordering.
8. Compare the attribution-guided and random deletion curves using area under the curve (AUC).

### Attribution-Guided Deletion

For each input, words are ranked according to:

```text
|attribution|
```

The most highly attributed word is removed first, followed by the next most highly attributed word, and so on.

The resulting target-class logits form the attribution-guided deletion curve.

### Random Baseline

The same deletion process is repeated with randomly shuffled word orders.

Multiple random seeds are used to estimate the mean and standard deviation of the random deletion curve.

This provides a baseline for determining whether attribution-guided deletion changes the model output more strongly than arbitrary deletion.

### Faithfulness Metrics

The current implementation measures cumulative target-logit drop:

```text
logit_drop = original_logit - deleted_logit
```

The area under the resulting deletion curve is then used as the summary statistic.

The evaluation also performs the same comparison using sigmoid probabilities rather than raw logits.

A larger AUC means that the target-class output changed more strongly over the deletion sequence. Because transformer predictions are nonlinear, the deletion curves are not required to be monotonic.

### Example Evaluation

An exploratory evaluation was performed on 10 manually selected IMDb sentiment examples.

The results were:

| Metric                    | Attribution-guided | Random baseline |
| ------------------------- | -----------------: | --------------: |
| Mean logit-drop AUC       |             2.1604 |          1.4099 |
| Mean probability-drop AUC |             0.3718 |          0.2390 |

Attribution-guided deletion produced a higher AUC than the mean random baseline on 9 of the 10 examples for both the logit- and probability-based evaluations.

These results are exploratory and should not be interpreted as a statistically representative benchmark because the evaluation set contains only 10 manually selected examples.

### Completeness

Integrated Gradients completeness was monitored by comparing:

```text
sum(attributions)
```

with:

```text
model(input) - model(baseline)
```

The resulting completeness error is recorded for each example.

For the current 10-example evaluation, the mean absolute completeness error was approximately:

```text
0.00824
```

### Limitations

The current evaluation has several limitations:

* The evaluation set contains only 10 manually selected examples.
* The examples are not intended to represent the full IMDb dataset.
* Deletion curves can be non-monotonic because transformer models are nonlinear.
* Raw logit differences can behave differently from probability changes.
* A higher deletion AUC does not by itself establish universal attribution faithfulness.
* Additional attribution methods and perturbation baselines should be evaluated.
* Larger and more representative evaluation datasets are needed for stronger conclusions.

## Robustness Analysis

The project also contains experiments for investigating how model predictions behave under changes to the input.

The current robustness work is available in:

```text
notebooks/robustness.ipynb
```

The robustness experiments are intended to complement attribution analysis by examining whether explanations and predictions remain meaningful when inputs are modified.

## Evaluation API

The reusable evaluation functionality is implemented in:

```text
xray/evaluation.py
```

The module provides functionality for:

* Predicting target-class logits
* Generating attribution-guided deletion curves
* Generating random deletion curves
* Repeating random deletion experiments across multiple seeds
* Summarizing random baselines
* Computing logit-based faithfulness metrics
* Computing probability-based faithfulness metrics
* Running the complete example-level evaluation pipeline

The module also exposes a defined public API through `__all__` and uses typed return structures for the main evaluation results.

## Testing

The project includes automated tests for the evaluation utilities.

Run the test suite from the project root:

```bash
pytest
```

The current test suite covers:

* WordPiece token grouping
* Input deletion behavior
* Logit-to-probability conversion
* Random curve summarization
* Repeated random faithfulness evaluation
* Probability-based faithfulness evaluation

## Installation

The project uses a standard Python package configuration through `pyproject.toml`.

Install the project in editable mode:

```bash
pip install -e .
```

Then verify that the package can be imported:

```bash
python -c "import xray; print(xray.__file__)"
```

## Research Direction

The project is being developed as a research-oriented toolkit rather than a single-model application.

Potential future directions include:

* Additional attribution methods
* Attention-based explanation analysis
* Attribution stability analysis
* More systematic robustness evaluation
* Uncertainty estimation
* Calibration analysis
* Mechanistic interpretability
* Activation patching
* Attention-head analysis
* Larger-scale evaluation datasets
* Additional perturbation-based faithfulness metrics

These areas are exploratory and may be implemented incrementally as the toolkit develops.

## Design Philosophy

X-Ray follows three principles:

### 1. Interpretability should be measurable

An explanation should not only be visualized. Its relationship with model behavior should also be evaluated.

### 2. Evaluation should include baselines

Attribution-guided perturbation is compared against random perturbation rather than being evaluated in isolation.

### 3. Experimental claims should match the evidence

Small exploratory experiments are treated as diagnostics rather than definitive evidence of general model behavior.

## Status

The project currently contains a working transformer interpretation pipeline, attribution faithfulness evaluation, robustness experiments, and automated evaluation tests.

The toolkit remains an ongoing research project.
