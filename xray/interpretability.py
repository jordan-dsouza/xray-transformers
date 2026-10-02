from typing import Any, TypedDict

import torch
from captum.attr import IntegratedGradients


class AttributionResult(TypedDict):
    tokens: list[str]
    token_attributions: torch.Tensor
    convergence_delta: torch.Tensor
    input_output: torch.Tensor
    baseline_output: torch.Tensor
    total_attribution: torch.Tensor
    completeness_error: torch.Tensor


class DistilBertInterpreter:
    """Integrated Gradients interpreter for DistilBERT classifiers."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.eval()
        self.ig = IntegratedGradients(self.forward_func)

    def forward_func(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model using input embeddings instead of token IDs."""

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        return outputs.logits

    def aggregate_tokens(
        self,
        tokens: list[str],
        attributions: torch.Tensor,
    ) -> tuple[list[str], list[float]]:
        """
        Merge WordPiece tokens into complete words.

        Special tokens are excluded.
        """

        words: list[str] = []
        scores: list[float] = []

        current_word = None
        current_score = 0.0

        for token, score in zip(tokens, attributions):
            score = float(score)

            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if not token.startswith("##"):
                if current_word is not None:
                    words.append(current_word)
                    scores.append(current_score)

                current_word = token
                current_score = score

            elif current_word is not None:
                current_word += token[2:]
                current_score += score

        if current_word is not None:
            words.append(current_word)
            scores.append(current_score)

        return words, scores

    def attribute(
        self,
        text: str,
        target_label: int | None = None,
    ) -> AttributionResult:
        """
        Calculate Integrated Gradients attribution scores for an input text.
        """

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=False,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        embeddings = self.model.distilbert.embeddings(input_ids)

        # Use PAD-token embeddings as the Integrated Gradients baseline.
        pad_token_id = self.tokenizer.pad_token_id

        baseline_ids = torch.full_like(
            input_ids,
            pad_token_id,
        )

        baseline = self.model.distilbert.embeddings(
            baseline_ids
        )

        # Explain the model's predicted class when no target is provided.
        if target_label is None:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                target_label = torch.argmax(
                    outputs.logits,
                    dim=1,
                ).item()

        with torch.no_grad():
            input_output = self.forward_func(
                embeddings,
                attention_mask,
            )[0, target_label]

            baseline_output = self.forward_func(
                baseline,
                attention_mask,
            )[0, target_label]

        attributions, delta = self.ig.attribute(
            inputs=embeddings,
            baselines=baseline,
            additional_forward_args=(attention_mask,),
            target=target_label,
            return_convergence_delta=True,
            n_steps=1000,
        )

        token_attributions = attributions.sum(
            dim=-1
        ).squeeze(0)

        total_attribution = token_attributions.sum()

        output_difference = input_output - baseline_output

        completeness_error = (
            total_attribution - output_difference
        )

        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0)
        )

        return {
            "tokens": tokens,
            "token_attributions": token_attributions.detach().cpu(),
            "convergence_delta": delta.detach().cpu(),
            "input_output": input_output.detach().cpu(),
            "baseline_output": baseline_output.detach().cpu(),
            "total_attribution": total_attribution.detach().cpu(),
            "completeness_error": completeness_error.detach().cpu(),
        }

    def visualize_attributions(
        self,
        words: list[str],
        scores: list[float],
    ) -> None:
        """
        Print word-level attributions as a simple text visualization.

        Positive scores support the prediction.
        Negative scores oppose the prediction.
        """

        print("\nATTRIBUTION VISUALIZATION")
        print("-" * 50)

        for word, score in zip(words, scores):
            score = float(score)

            bar_length = min(
                int(abs(score) * 20),
                30,
            )

            if score >= 0:
                bar = "+" * bar_length
            else:
                bar = "-" * bar_length

            print(
                f"{word:15} {score:+.4f}  {bar}"
            )