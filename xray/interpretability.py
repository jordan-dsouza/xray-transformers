import torch
from captum.attr import IntegratedGradients

class DistilBertInterpreter:
    def __init__(self, model, tokenizer, device):
        # Store trained model, tokenizer and device:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Switch to evaluation mode:
        self.model.eval()

        # Integrated Gradients uses forward function:
        self.ig = IntegratedGradients(self.forward_func)

    def forward_func(self, inputs_embeds, attention_mask):
        """
        Forward func uses embedding instead of token IDs, allowing Captum to calculate gradients
        """

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        # Class prediction scores:
        return outputs.logits

    def attribute(self, text, target_label=None):
        """
        Calculate importance scores for each token
        """

        # Convert text into token IDs and attention mask:
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # Move inputs to device:
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Convert token IDs into embeddings:
        embeddings = self.model.distilbert.embeddings(input_ids)

        # If no target class, explain model prediction:
        if target_label is None:
            with torch.no_grad():
                
                # Get model prediction:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Select predicted class:
                target_label = torch.argmax(
                    outputs.logits,
                    dim=1
                ).item()
        
        # Calculate INTEGRATED GRADIENTS:
        attributions, delta = self.ig.attribute(
            inputs=embeddings,
            additional_forward_args=(attention_mask,),
            target=target_label,
            return_convergence_delta=True
        )

        # Combine attribution values across embedded dimensions:
        attributions = attributions.sum(dim=-1).squeeze(0)

        # Convert token IDs back into readable tokens:
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0)
        )

        # Return tokens, importance scores and convergance info:
        return tokens, attributions.detach().cpu(), delta