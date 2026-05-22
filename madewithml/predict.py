import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import ray
import typer
from ray.train import Checkpoint, Result
from typing_extensions import Annotated

from madewithml.config import logger, mlflow
from madewithml.data import CustomPreprocessor
from madewithml.models import FinetunedLLM
from madewithml.utils import collate_fn

# Initialize Typer CLI app
app = typer.Typer()


def decode(indices: list[int], index_to_class: dict[int, str]) -> list[str]:
    """Decode numerical predictions back to human-readable text labels."""
    return [index_to_class[index] for index in indices]


def format_prob(prob: np.ndarray, index_to_class: dict[int, str]) -> dict[str, float]:
    """
    Format the raw probability array into a clean dictionary.
    Crucially, this converts numpy floats to native Python floats so we 
    can serialize to JSON natively without needing third-party encoders (great for Docker).
    """
    return {index_to_class[i]: float(item) for i, item in enumerate(prob)}


class TorchPredictor:
    """
    A custom predictor class designed to work seamlessly with Ray Data's map_batches.
    It holds both the text preprocessor and the neural network in memory.
    """
    def __init__(self, preprocessor: CustomPreprocessor, model: FinetunedLLM):
        self.preprocessor = preprocessor
        self.model = model
        # Ensure the model is strictly in evaluation mode (turns off dropout)
        self.model.eval()

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        This method makes the class callable. Ray Data will pass batches of data here.
        collate_fn handles converting the numpy arrays to padded PyTorch tensors.
        """
        results = self.model.predict(collate_fn(batch))
        return {"output": results}

    def predict_proba(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Predicts probability distributions instead of hard classes."""
        results = self.model.predict_proba(collate_fn(batch))
        return {"output": results}

    def get_preprocessor(self) -> CustomPreprocessor:
        return self.preprocessor

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint) -> "TorchPredictor":
        """
        Factory method to reconstruct the entire prediction pipeline (processor + model)
        directly from a saved Ray checkpoint on disk.
        """
        # Modern Ray syntax for safely accessing checkpoint files
        with checkpoint.as_directory() as checkpoint_dir:
            metadata = checkpoint.get_metadata()
            
            # 1. Rebuild the preprocessor with the saved label mappings
            preprocessor = CustomPreprocessor(class_to_index=metadata["class_to_index"])
            
            # 2. Rebuild the model architecture and load the learned weights
            model = FinetunedLLM.load(
                Path(checkpoint_dir, "args.json"), 
                Path(checkpoint_dir, "model.pt")
            )
            
        return cls(preprocessor=preprocessor, model=model)


def predict_proba_dataset(
    ds: ray.data.Dataset,
    predictor: TorchPredictor,
) -> list[dict[str, Any]]:
    """
    Runs the full inference pipeline (preprocessing -> prediction -> formatting) 
    on a Ray Dataset.
    """
    preprocessor = predictor.get_preprocessor()
    
    # 1. Apply the exact same text cleaning and tokenization used during training
    preprocessed_ds = preprocessor.transform(ds)
    
    # 2. Run the model over the dataset in batches for parallel efficiency
    outputs = preprocessed_ds.map_batches(predictor.predict_proba)
    
    # 3. Extract the results, determine the highest probability tag, and format
    y_prob = np.array([d["output"] for d in outputs.take_all()])
    results = []
    
    for prob in y_prob:
        tag = preprocessor.index_to_class[prob.argmax()]
        results.append({
            "prediction": tag, 
            "probabilities": format_prob(prob, preprocessor.index_to_class)
        })
        
    return results


@app.command()
def get_best_run_id(experiment_name: str = "", metric: str = "", mode: str = "ASC") -> str:
    """
    Queries the MLflow tracking server to find the absolute best model 
    based on a specific metric (e.g., lowest val_loss).
    """
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} {mode}"],
    )
    
    if sorted_runs.empty:
        raise ValueError(f"No runs found for experiment: {experiment_name}")
        
    run_id = sorted_runs.iloc[0].run_id
    print(f"Best Run ID: {run_id}")
    return run_id


def get_best_checkpoint(run_id: str) -> Checkpoint:
    """
    Retrieves the actual Ray Checkpoint object from disk using the MLflow run ID.
    """
    # Ask MLflow where it saved the artifacts for this specific run
    run_info = mlflow.get_run(run_id).info
    artifact_dir = urlparse(run_info.artifact_uri).path
    
    # Load the Result object from the trial directory to extract the checkpoint
    result = Result.from_path(artifact_dir)
    return result.best_checkpoints[0][0]


@app.command()
def predict(
    run_id: Annotated[str, typer.Option(help="id of the specific run to load from")] = None,
    title: Annotated[str, typer.Option(help="project title")] = "",
    description: Annotated[str, typer.Option(help="project description")] = "",
) -> list[dict[str, Any]]:
    """
    End-to-end inference CLI command.
    Takes a raw title and description, loads the best model, and predicts the category.
    """
    # 1. Locate and load the best saved model and preprocessor
    best_checkpoint = get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)

    # 2. Package the raw string inputs into a Ray Dataset
    sample_ds = ray.data.from_items([{
        "title": title, 
        "description": description, 
        "tag": "other"  # Dummy target since we don't know the real label yet
    }])
    
    # 3. Run the prediction pipeline
    results = predict_proba_dataset(ds=sample_ds, predictor=predictor)
    
    # 4. Log the results nicely (No custom NumpyEncoder needed anymore!)
    logger.info(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    app()