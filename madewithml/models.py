import json
from pathlib import Path
from typing import Any

import torch
import numpy as np  
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class FinetunedLLM(nn.Module):
    """
    A custom PyTorch neural network that fine-tunes a pre-trained Large Language Model (BERT) 
    for multi-class text classification.
    """

    def __init__(self, llm: BertModel, dropout_p: float, embedding_dim: int, num_classes: int):
        super().__init__()
        
        # 1. The Pre-trained LLM (Feature Extractor)
        # We pass in a pre-loaded BERT model. This turns our raw text tokens into rich, dense vectors.
        self.llm = llm
        
        # Store architecture parameters so we can save/load the model exactly as it was built
        self.dropout_p = dropout_p
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # 2. Dropout Layer (Regularization)
        # Randomly zeroes out some of the neural network's connections during training.
        # This prevents the model from memorizing the training data (overfitting).
        self.dropout = nn.Dropout(dropout_p)
        
        # 3. The Classification Head (Fully Connected Layer)
        # Takes the high-dimensional embeddings from BERT (e.g., 768 dimensions) 
        # and condenses them down to our specific number of target tags/classes.
        self.fc1 = nn.Linear(embedding_dim, num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        The forward pass defines how data flows through the network from input to output.
        """
        # Extract the token IDs and attention masks from the incoming batch
        ids, masks = batch["ids"], batch["masks"]
        
        # Pass inputs through BERT.
        # 'seq' is the hidden state for every single token (we don't need this for simple classification).
        # 'pool' is the [CLS] token's representation, which acts as an aggregate summary of the entire sentence.
        seq, pool = self.llm(input_ids=ids, attention_mask=masks)
        
        # Apply dropout to the sentence summary for regularization
        z = self.dropout(pool)
        
        # Pass the summary through our final linear layer to get the raw scores (logits) for each class
        z = self.fc1(z)
        
        return z

    @torch.inference_mode()
    def predict(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        """
        Predicts the single most likely class for a batch of inputs.
        Uses @torch.inference_mode() to disable gradient tracking, making it much faster and using less memory.
        """
        self.eval()  # Put the model in evaluation mode (turns off Dropout)
        z = self(batch)  # Get the raw logits from the forward pass
        
        # Find the index of the highest score across the class dimension (dim=1)
        y_pred = torch.argmax(z, dim=1).cpu().numpy()
        
        return y_pred

    @torch.inference_mode()
    def predict_proba(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        """
        Predicts the probability distribution across all classes.
        Useful when you want to know how confident the model is (e.g., setting a confidence threshold).
        """
        self.eval()
        z = self(batch)
        
        # Apply the softmax function to convert raw logits into percentages that sum to 1.0
        y_probs = F.softmax(z, dim=1).cpu().numpy()
        
        return y_probs

    def save(self, dp: str | Path) -> None:
        """
        Saves the model's architecture arguments and its learned weights to a directory.
        """
        dp = Path(dp)
        dp.mkdir(parents=True, exist_ok=True)
        
        # 1. Save the architecture arguments so we know how to rebuild the PyTorch class
        with open(dp / "args.json", "w") as fp:
            contents = {
                "dropout_p": self.dropout_p,
                "embedding_dim": self.embedding_dim,
                "num_classes": self.num_classes,
            }
            json.dump(contents, fp, indent=4, sort_keys=False)
            
        # 2. Save the actual learned weights (the state dictionary)
        torch.save(self.state_dict(), dp / "model.pt")

    @classmethod
    def load(cls, args_fp: str | Path, state_dict_fp: str | Path) -> "FinetunedLLM":
        """
        A factory method to reconstruct the model from saved files.
        """
        # Load the architecture parameters
        with open(args_fp, "r") as fp:
            kwargs = json.load(fp=fp)
            
        # Initialize the base BERT model
        llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
        
        # Reconstruct our custom model using the saved parameters
        model = cls(llm=llm, **kwargs)
        
        # Inject the saved weights into the reconstructed model.
        # map_location="cpu" ensures we can load a model trained on a GPU onto a standard CPU environment.
        model.load_state_dict(torch.load(state_dict_fp, map_location=torch.device("cpu")))
        
        return model