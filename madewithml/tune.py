import datetime
import json

import ray
import typer
from ray import tune
from ray.train import (
    CheckpointConfig,
    DataConfig,
    RunConfig,
    ScalingConfig,
)
# Modern Ray API: integrations moved from ray.air to ray.train
from ray.train.mlflow import MLflowLoggerCallback
from ray.train.torch import TorchTrainer
from ray.tune import Tuner
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from typing_extensions import Annotated

from madewithml import data, train, utils
from madewithml.config import MLFLOW_TRACKING_URI, logger, settings

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def tune_models(
    experiment_name: Annotated[str, typer.Option(help="name of the experiment")] = "tune_default",
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset")] = "",
    initial_params: Annotated[str, typer.Option(help="initial config JSON string")] = "{}",
    num_workers: Annotated[int, typer.Option(help="number of workers to use for training")] = 1,
    cpu_per_worker: Annotated[int, typer.Option(help="number of CPUs to use per worker")] = 1,
    gpu_per_worker: Annotated[int, typer.Option(help="number of GPUs to use per worker")] = 0,
    num_runs: Annotated[int, typer.Option(help="number of HPO trials to run")] = 1,
    num_samples: Annotated[int | None, typer.Option(help="number of samples to use")] = None,
    num_epochs: Annotated[int, typer.Option(help="number of epochs to train for")] = 1,
    batch_size: Annotated[int, typer.Option(help="number of samples per batch")] = 256,
    results_fp: Annotated[str | None, typer.Option(help="filepath to save results to")] = None,
) -> ray.tune.ResultGrid:
    """Hyperparameter tuning experiment using Ray Tune."""
    
    # 1. Setup Base Configuration
    utils.set_seeds()
    train_loop_config = {
        "num_samples": num_samples,
        "num_epochs": num_epochs,
        "batch_size": batch_size
    }

    # Define hardware resources per trial
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(gpu_per_worker),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker},
    )

    # 2. Prepare the Dataset
    ds = data.load_data(dataset_loc=dataset_loc, num_samples=train_loop_config.get("num_samples"))
    train_ds, val_ds = data.stratify_split(ds, stratify="tag", test_size=0.2)
    tags = train_ds.unique(column="tag")
    train_loop_config["num_classes"] = len(tags)

    # Modern Ray DataConfig replacing legacy DatasetConfig dicts
    dataset_config = DataConfig(
        execution_options=ray.data.ExecutionOptions(preserve_order=True)
    )

    # Preprocess and materialize into memory before tuning starts
    preprocessor = data.CustomPreprocessor()
    preprocessor = preprocessor.fit(train_ds)
    train_ds = preprocessor.transform(train_ds).materialize()
    val_ds = preprocessor.transform(val_ds).materialize()

    # 3. Define the Base Trainer
    # This is the exact same trainer used in train.py, but Tune will wrap it and mutate its config
    trainer = TorchTrainer(
        train_loop_per_worker=train.train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
        metadata={"class_to_index": preprocessor.class_to_index},
    )

    # 4. Configure MLflow and Storage
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=experiment_name,
        save_artifact=True,
    )
    
    # Use dynamic pathing for Docker volume compatibility
    storage_path = str(settings.efs_dir)
    run_config = RunConfig(
        callbacks=[mlflow_callback], 
        checkpoint_config=checkpoint_config, 
        storage_path=storage_path,
        name=experiment_name
    )

    # 5. Define the Search Algorithm (HyperOpt)
    # HyperOpt uses Bayesian Optimization to guess the best parameters based on past trials
    initial_params_dict = json.loads(initial_params)
    search_alg = HyperOptSearch(points_to_evaluate=[initial_params_dict] if initial_params_dict else None)
    # Limit concurrent trials so we don't overwhelm the machine/cluster
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)

    # Define the parameter space we want to explore
    param_space = {
        "train_loop_config": {
            "dropout_p": tune.uniform(0.3, 0.9),
            "lr": tune.loguniform(1e-5, 5e-4),
            "lr_factor": tune.uniform(0.1, 0.9),
            "lr_patience": tune.uniform(1, 10),
        }
    }

    # 6. Define the Scheduler (ASHA)
    # ASHA (Asynchronous Successive Halving) monitors trials. If a trial is performing poorly 
    # early on, ASHA kills it immediately to save compute time for more promising configurations.
    scheduler = AsyncHyperBandScheduler(
        max_t=train_loop_config["num_epochs"],  # Maximum epochs a trial can run
        grace_period=1,                         # Minimum epochs before a trial can be killed
    )

    tune_config = tune.TuneConfig(
        metric="val_loss",
        mode="min",
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=num_runs,
    )

    # 7. Execute Tuning
    tuner = Tuner(
        trainable=trainer,
        run_config=run_config,
        param_space=param_space,
        tune_config=tune_config,
    )

    results = tuner.fit()
    
    # Extract best trial safely
    best_trial = results.get_best_result(metric="val_loss", mode="min")
    metrics_dict = best_trial.metrics_dataframe.to_dict() if not best_trial.metrics_dataframe.empty else {}
    
    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": utils.get_run_id(experiment_name=experiment_name, trial_id=best_trial.metrics.get("trial_id", "")),
        "params": best_trial.config.get("train_loop_config", {}),
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