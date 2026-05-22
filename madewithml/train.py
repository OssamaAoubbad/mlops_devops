import datetime
import json
import os
import tempfile

import mlflow
import numpy as np
import ray
import ray.train as train
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from ray.data import Dataset
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    DataConfig,
    RunConfig,
    ScalingConfig,
)
from ray.train.torch import TorchTrainer
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import BertModel
from typing_extensions import Annotated

from madewithml import data, utils
from madewithml.config import MLFLOW_TRACKING_URI, logger, settings
from madewithml.models import FinetunedLLM

# Initialize Typer CLI app
app = typer.Typer()


def train_step(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Executes one full pass (epoch) over the training dataset."""
    model.train()  # Turn on Dropout and layer tracking
    loss = 0.0
    
    # Iterate through Ray Data in native PyTorch tensors
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()  # 1. Clear old gradients from the last step
        z = model(batch)       # 2. Forward pass: compute predictions
        
        # 3. Convert integer labels to one-hot encoded vectors for BCEWithLogitsLoss
        targets = F.one_hot(batch["targets"], num_classes=num_classes).float()
        
        J = loss_fn(z, targets) # 4. Calculate the error (loss)
        J.backward()            # 5. Backward pass: calculate gradients
        optimizer.step()        # 6. Update the model weights
        
        # Calculate cumulative moving average of the loss
        loss += (J.detach().item() - loss) / (i + 1)
        
    return float(loss)


def eval_step(
    ds: Dataset, 
    batch_size: int, 
    model: nn.Module, 
    num_classes: int, 
    loss_fn: nn.modules.loss._WeightedLoss
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluates the model on the validation set without updating weights."""
    model.eval()  # Turn off Dropout
    loss = 0.0
    y_trues, y_preds = [], []
    
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    
    # torch.inference_mode() is a faster, stricter version of torch.no_grad()
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            targets = F.one_hot(batch["targets"], num_classes=num_classes).float()
            
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
            
    return float(loss), np.vstack(y_trues), np.vstack(y_preds)


def train_loop_per_worker(config: dict[str, float | int | str]) -> None:
    """
    The core loop executed by EVERY distributed worker (CPU or GPU).
    Ray Train handles copying this across your cluster/container automatically.
    """
    # Hyperparameters successfully unpacked from our updated config_dict
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]
    experiment_name = config["experiment_name"]

    # ====================================================================
    # MLFLOW INTEGRATION: Initialize tracking ONLY on the master worker
    # ====================================================================
    is_master = train.get_context().get_world_rank() == 0
    if is_master:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.log_params(config)

    # Get data specifically assigned to this worker
    utils.set_seeds()
    train_ds = train.get_dataset_shard("train")
    val_ds = train.get_dataset_shard("val")

    # Initialize the architecture
    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = FinetunedLLM(llm=llm, dropout_p=dropout_p, embedding_dim=llm.config.hidden_size, num_classes=num_classes)
    
    # Wrap model for distributed training natively through Ray
    model = train.torch.prepare_model(model)

    # Optimization setup
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)

    # Calculate batch size dynamically based on cluster size
    num_workers = train.get_context().get_world_size()
    batch_size_per_worker = batch_size // num_workers

    for epoch in range(num_epochs):
        # 1. Train & Evaluate
        train_loss = train_step(train_ds, batch_size_per_worker, model, num_classes, loss_fn, optimizer)
        val_loss, _, _ = eval_step(val_ds, batch_size_per_worker, model, num_classes, loss_fn)
        
        # 2. Adjust Learning Rate based on validation plateau
        scheduler.step(val_loss)

        # ====================================================================
        # MLFLOW INTEGRATION: Log metrics per epoch
        # ====================================================================
        if is_master:
            mlflow.log_metrics({
                "train_loss": train_loss, 
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"]
            }, step=epoch)

        # 3. Save Checkpoints and Report Metrics to Ray
        with tempfile.TemporaryDirectory() as dp:
            # Handle standard vs DistributedDataParallel (DDP) wrappers
            if isinstance(model, DistributedDataParallel):
                model.module.save(dp=dp)
            else:
                model.save(dp=dp)
                
            metrics = {
                "epoch": epoch, 
                "lr": optimizer.param_groups[0]["lr"], 
                "train_loss": train_loss, 
                "val_loss": val_loss
            }
            
            # Modern Checkpoint API
            checkpoint = Checkpoint.from_directory(dp)
            train.report(metrics, checkpoint=checkpoint)

    # ====================================================================
    # MLFLOW INTEGRATION: Close the run safely
    # ====================================================================
    if is_master:
        mlflow.end_run()


@app.command()
def train_model(
    experiment_name: Annotated[str, typer.Option(help="name of the experiment")] = "default",
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset")] = "",
    train_loop_config: Annotated[str, typer.Option(help="JSON string of arguments to use for training")] = "{}",
    num_workers: Annotated[int, typer.Option(help="number of workers to use for training")] = 1,
    cpu_per_worker: Annotated[int, typer.Option(help="number of CPUs to use per worker")] = 1,
    gpu_per_worker: Annotated[int, typer.Option(help="number of GPUs to use per worker")] = 0,
    num_samples: Annotated[int | None, typer.Option(help="number of samples to use")] = None,
    num_epochs: Annotated[int, typer.Option(help="number of epochs to train for")] = 1,
    batch_size: Annotated[int, typer.Option(help="number of samples per batch")] = 256,
    results_fp: Annotated[str | None, typer.Option(help="filepath to save results to")] = None,
) -> ray.train.Result:
    """Main function to launch the distributed training workload via Ray Train."""
    
    # Parse hyperparams from CLI
    config_dict = json.loads(train_loop_config)
    
    # Inject default model hyperparameters so the worker doesn't throw a KeyError
    config_dict.setdefault("dropout_p", 0.5)
    config_dict.setdefault("lr", 1e-4)
    config_dict.setdefault("lr_factor", 0.8)
    config_dict.setdefault("lr_patience", 3)
    
    # Inject CLI args (Including experiment_name for MLflow)
    config_dict.update({
        "num_samples": num_samples,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "experiment_name": experiment_name
    })

    # 1. Define hardware resources for the cluster
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(gpu_per_worker),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker},
    )

    # 2. Define Checkpoint configurations
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,  # Only keep the absolute best model to save disk space
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    # Use settings.efs_dir for cross-environment compatibility (Docker vs Local)
    storage_path = str(settings.efs_dir)
    run_config = RunConfig(
        checkpoint_config=checkpoint_config, 
        storage_path=storage_path, 
        name=experiment_name
    )

    # 3. Prepare the Dataset
    ds = data.load_data(dataset_loc=dataset_loc, num_samples=config_dict["num_samples"])
    train_ds, val_ds = data.stratify_split(ds, stratify="tag", test_size=0.2)
    tags = train_ds.unique(column="tag")
    config_dict["num_classes"] = len(tags)

    # Ensure deterministic block order for reproducible distributed training
    dataset_config = DataConfig(
        execution_options=ray.data.ExecutionOptions(preserve_order=True)
    )

    # Preprocess and materialize in memory
    preprocessor = data.CustomPreprocessor()
    preprocessor = preprocessor.fit(train_ds)
    train_ds = preprocessor.transform(train_ds).materialize()
    val_ds = preprocessor.transform(val_ds).materialize()

    # 4. Initialize and launch the TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config_dict,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
    )

    results = trainer.fit()
    
    # 5. Format outputs & Fetch MLflow ID
    metrics_dict = results.metrics_dataframe.to_dict() if not results.metrics_dataframe.empty else {}
    
    # Automatically search the local MLflow registry for the exact Run ID
    run_id = "unknown"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
    
    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": run_id,
        "params": config_dict,
        "metrics": utils.dict_to_list(metrics_dict, keys=["epoch", "train_loss", "val_loss"]) if metrics_dict else [],
    }
    
    logger.info(json.dumps(d, indent=2))
    
    if results_fp:
        utils.save_dict(d, results_fp)
        
    return results


if __name__ == "__main__":
    # Ensure a clean slate for Ray initialization
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()