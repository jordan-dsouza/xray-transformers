# xray-transformers
X-Ray: A Unified Interpretability &amp; Uncertainty Toolkit for Transformer Models
# X-Ray: Interpretability & Uncertainty for Transformers

X-Ray is an open-source research toolkit for analyzing transformer-based models through:

- Feature attribution (gradient and attention - based)
- Mechanistic interpretability (activation patching, head analysis)
- Uncertainty estimation (ensembles, MC dropout)
- Robustness and failure mode discovery

## Motivation

Transformer models achieve strong performance but remain opaque, overconfident, and fragile.
This project aims to connect **internal model mechanisms** with **human-interpretable explanations**
and **quantified uncertainty**.

## Scope

- Focused on text classification tasks
- Emphasis on interpretability and trust, not SOTA accuracy
- Research-oriented, reproducible experiments

## Roadmap

- [ ] Baseline fine-tuning (DistilBERT on IMDb)
- [ ] Integrated Gradients & attention explanations
- [ ] Activation patching & head-level causal analysis
- [ ] Uncertainty estimation & calibration
- [ ] Explanation stability & failure mode analysis
