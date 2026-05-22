# tests/test_models.py
import pytest
import torch
from transformers import BertModel
from madewithml.models import FinetunedLLM

@pytest.fixture
def mock_batch():
    """Create a fake batch of tokenized data to pass through the model."""
    batch_size = 4
    seq_length = 10
    return {
        "ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "masks": torch.ones(batch_size, seq_length, dtype=torch.int32),
        "targets": torch.randint(0, 3, (batch_size,))
    }

def test_model_forward_pass(mock_batch):
    """Ensure the neural network outputs the correct tensor shapes."""
    num_classes = 3
    embedding_dim = 768
    
    # Initialize the base BERT model and our custom wrapper
    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = FinetunedLLM(llm=llm, dropout_p=0.5, embedding_dim=embedding_dim, num_classes=num_classes)
    
    # Run the fake data through the forward pass
    logits = model(mock_batch)
    
    # We expect the output shape to be [Batch Size, Number of Classes]
    assert logits.shape == (4, num_classes)
    assert isinstance(logits, torch.Tensor)