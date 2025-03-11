"""
Tune hyperparameters using Optuna.
"""

import json
import mlflow
import optuna
import pandas as pd
import sys

from loguru import logger

sys.path.append(".")

from src.dataset import download  # noqa:E402
from src.modeling.train import tune_hp  # noqa:E402


def main():
    with open("configs/data.json") as f:
        data_config = json.load(f)
    logger.info("Downloading data")
    download(data_config["url"])
    df_train = pd.read_csv(data_config["train_path"])
    df_train = df_train.dropna(subset=["target"])
    df_train = df_train.reset_index(drop=True)
    with open("configs/optuna.json") as f:
        tune_config = json.load(f)
    mlflow.set_experiment("LGBM Tuning")
    with mlflow.start_run():
        study = tune_hp(
            df_train,
            tune_config["lgb_params_base"],
            tune_config["lgb_params_tune"],
            tune_config["pruner_params"],
            tune_config["n_trials"],
            data_config["split_day"],
        )
    fig = optuna.visualization.plot_optimization_history(study)
    mlflow.log_figure(fig, "optuna.html")
    best_trial = study.best_trial
    mlflow.log_metric("MAE", best_trial.value)
    mlflow.log_params(best_trial.params)


if __name__ == "__main__":
    main()
