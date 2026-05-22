import datetime
import json
from typing import Any

import numpy as np
import pandas as pd
import ray
import typer
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slicing_function
from typing_extensions import Annotated

from madewithml import predict, utils
from madewithml.config import logger
from madewithml.predict import TorchPredictor

# Initialize Typer CLI app
app = typer.Typer()


def get_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Get overall performance metrics using native Python types for JSON compatibility."""
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    return {
        "precision": float(metrics[0]),
        "recall": float(metrics[1]),
        "f1": float(metrics[2]),
        "num_samples": int(len(y_true)),
    }


def get_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_to_index: dict[str, int]
) -> dict[str, dict[str, float | int]]:
    """Get per-class performance metrics, sorted by F1 score."""
    per_class_metrics = {}
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    for i, _class in enumerate(class_to_index):
        per_class_metrics[_class] = {
            "precision": float(metrics[0][i]),
            "recall": float(metrics[1][i]),
            "f1": float(metrics[2][i]),
            "num_samples": int(metrics[3][i]),
        }
        
    # Modern Python dicts maintain order natively. No OrderedDict needed.
    return dict(sorted(per_class_metrics.items(), key=lambda item: item[1]["f1"], reverse=True))


@slicing_function()
def nlp_llm(x: pd.Series) -> bool:
    """Snorkel slice: NLP projects that use LLMs."""
    nlp_project = "natural-language-processing" in x.tag
    llm_terms = ["transformer", "llm", "bert"]
    llm_project = any(s.lower() in x.text.lower() for s in llm_terms)
    return nlp_project and llm_project


@slicing_function()
def short_text(x: pd.Series) -> bool:
    """Snorkel slice: Projects with short titles and descriptions."""
    return len(str(x.text).split()) < 8


def get_slice_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, ds: ray.data.Dataset
) -> dict[str, dict[str, float | int]]:
    """Get performance metrics for specifically defined Snorkel slices."""
    slice_metrics = {}
    
    # Convert Ray dataset to Pandas specifically for Snorkel applier
    df = ds.to_pandas()
    df["text"] = df["title"] + " " + df["description"]
    slices = PandasSFApplier([nlp_llm, short_text]).apply(df)
    
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        
        # Use numpy's optimized .sum() over built-in sum()
        if mask.sum() > 0:
            metrics = precision_recall_fscore_support(y_true[mask], y_pred[mask], average="micro")
            slice_metrics[slice_name] = {
                "precision": float(metrics[0]),
                "recall": float(metrics[1]),
                "f1": float(metrics[2]),
                "num_samples": int(mask.sum()),
            }
            
    return slice_metrics


@app.command()
def evaluate(
    run_id: Annotated[str, typer.Option(help="id of the specific run to load from")],
    dataset_loc: Annotated[str, typer.Option(help="dataset (with labels) to evaluate on")],
    results_fp: Annotated[str | None, typer.Option(help="location to save evaluation results to")] = None,
) -> dict[str, Any]:
    """Evaluate the trained model on the holdout dataset."""
    
    # 1. Load the dataset and the best model components
    ds = ray.data.read_csv(dataset_loc)
    best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)

    # 2. Extract ground truth labels (y_true)
    preprocessor = predictor.get_preprocessor()
    preprocessed_ds = preprocessor.transform(ds)
    values = preprocessed_ds.select_columns(cols=["targets"]).take_all()
    y_true = np.stack([item["targets"] for item in values])

    # 3. Generate predictions (y_pred)
    predictions = preprocessed_ds.map_batches(predictor).take_all()
    y_pred = np.array([d["output"] for d in predictions])

    # 4. Calculate all metrics
    metrics = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": run_id,
        "overall": get_overall_metrics(y_true=y_true, y_pred=y_pred),
        "per_class": get_per_class_metrics(y_true=y_true, y_pred=y_pred, class_to_index=preprocessor.class_to_index),
        "slices": get_slice_metrics(y_true=y_true, y_pred=y_pred, ds=ds),
    }
    
    # 5. Log and optionally save
    logger.info(json.dumps(metrics, indent=2))
    
    if results_fp:
        utils.save_dict(d=metrics, path=results_fp)
        
    return metrics


if __name__ == "__main__":
    app()