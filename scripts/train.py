"""
Train a voting ensemble of models using K-fold.

This splitting aproach is nonstandard for time series problems but for this task it
empirically demonstrated the best performance in ensembling.
"""

import joblib as jb
import json
import mlflow
import pandas as pd
import sys

from loguru import logger

sys.path.append(".")

from src.dataset import download  # noqa:E402
from src.modeling.train import train_k_folds  # noqa:E402


def main():
    with open("configs/data.json") as f:
        data_config = json.load(f)
    logger.info("Downloading data")
    download(data_config["url"])
    df_train = pd.read_csv(data_config["train_path"])
    df_train = df_train.dropna(subset=["target"])
    df_train = df_train.reset_index(drop=True)
    with open("configs/model.json") as f:
        model_config = json.load(f)
    mlflow.set_experiment("LGBM Training")
    with mlflow.start_run():
        models, metrics = train_k_folds(
            df_train,
            data_config["num_folds"],
            data_config["folds_gap"],
            model_config["lgb_params"],
        )
        mlflow.log_params(model_config["lgb_params"])
        mlflow.log_metrics(metrics)
        run_ts = mlflow.active_run().info.start_time
    logger.info("Saving models")
    save_path = f"models/{run_ts}.jb"
    jb.dump(models, save_path)


if __name__ == "__main__":
    main()
