import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ray.train.torch import get_device

from madewithml.config import mlflow


def set_seeds(seed: int = 42) -> None:
    """Set seeds for reproducibility across NumPy, Python, and PyTorch."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Safely set standard cudnn flags for deterministic behavior on GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dict(path: str | Path) -> dict[str, Any]:
    """Load a dictionary from a JSON filepath."""
    with open(path, "r") as fp:
        d = json.load(fp)
    return d


def save_dict(d: dict[str, Any], path: str | Path, cls: Any = None, sortkeys: bool = False) -> None:
    """
    Save a dictionary to a specific location safely.
    Uses pathlib to automatically create missing parent directories (Great for Docker volumes).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")


def pad_array(arr: np.ndarray, dtype: Any = np.int32) -> np.ndarray:
    """
    Pad a 2D array with zeros until all rows are of the same length
    as the longest row in the batch. Crucial for dynamic batch sizes in Transformers.
    """
    max_len = max(len(row) for row in arr)
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype)
    
    for i, row in enumerate(arr):
        padded_arr[i][: len(row)] = row
        
    return padded_arr


def collate_fn(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """
    Convert a batch of raw numpy arrays (from Ray Data) into padded PyTorch tensors.
    Automatically places the tensors on the correct device (CPU/GPU) mapped by Ray.
    """
    batch["ids"] = pad_array(batch["ids"])
    batch["masks"] = pad_array(batch["masks"])
    
    dtypes = {"ids": torch.int32, "masks": torch.int32, "targets": torch.int64}
    tensor_batch = {}
    
    # get_device() automatically detects the specific GPU assigned to this Ray worker
    device = get_device()
    
    for key, array in batch.items():
        tensor_batch[key] = torch.as_tensor(array, dtype=dtypes[key], device=device)
        
    return tensor_batch


def get_run_id(experiment_name: str, trial_id: str) -> str:
    """
    Queries MLflow to get the unique run ID for a specific Ray trial ID.
    Used to bridge the gap between Ray Tune/Train and MLflow Tracking.
    """
    trial_name = f"TorchTrainer_{trial_id}"
    run = mlflow.search_runs(
        experiment_names=[experiment_name], 
        filter_string=f"tags.trial_name = '{trial_name}'"
    ).iloc[0]
    
    return run.run_id


def dict_to_list(data: dict[str, list[Any]], keys: list[str]) -> list[dict[str, Any]]:
    """
    Convert a column-oriented dictionary (like Pandas to_dict()) 
    into a row-oriented list of dictionaries (standard JSON array format).
    """
    list_of_dicts = []
    
    # Safety check to prevent IndexErrors if the incoming dataframe/dict was empty
    if not data or not keys or keys[0] not in data:
        return list_of_dicts
        
    for i in range(len(data[keys[0]])):
        new_dict = {key: data[key][i] for key in keys}
        list_of_dicts.append(new_dict)
        
    return list_of_dicts