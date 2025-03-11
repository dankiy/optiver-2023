import gc
import lightgbm as lgb
import optuna
import pandas as pd

from loguru import logger
from sklearn.metrics import mean_absolute_error

from ..dataset import split
from ..features import generate_all_features


def train_k_folds(
    df_train: pd.DataFrame,
    num_folds: int,
    gap: int,
    lgb_params: dict,
) -> list[lgb.LGBMRegressor]:
    """
    Trains a set of LightGBM models using K-fold cross-validation.

    This function divides the training data into K folds, trains a model for each fold,
    and evaluates its performance using the mean absolute error. The last model is then
    trained on the entire dataset.

    Parameters:
    -----------
    df_train : pd.DataFrame
        The training dataset, including features and target values.
    num_folds : int
        The number of folds for cross-validation.
    gap : int
        The gap (in days) between the training and validation periods.
    lgb_params : dict
        A dictionary of parameters to be passed to the LightGBM model.

    Returns:
    --------
    list[lgb.LGBMRegressor]
        A list of trained LightGBM models, including models trained on each fold
        and the final model trained on the entire dataset.
    dict
        A dictionary containing the evaluation scores (MAE) for each fold.
    """
    fold_size = len(df_train.date_id.unique()) // num_folds
    date_ids = df_train["date_id"].values

    logger.info("Preparing data")
    df_train_feats = generate_all_features(df_train, True)
    feature_columns = df_train_feats.columns

    models = []
    scores = {}

    logger.info("Training models")
    for i in range(num_folds):
        start = i * fold_size
        end = start + fold_size
        if i < num_folds - 1:  # No need to purge after the last fold
            purged_start = end - 2
            purged_end = end + gap + 2
            train_indices = (date_ids >= start) & (date_ids < purged_start) | (
                date_ids > purged_end
            )
        else:
            train_indices = (date_ids >= start) & (date_ids < end)

        test_indices = (date_ids >= end) & (date_ids < end + fold_size)

        gc.collect()

        df_fold_train = df_train_feats[train_indices]
        df_fold_train_target = df_train["target"][train_indices]
        df_fold_valid = df_train_feats[test_indices]
        df_fold_valid_target = df_train["target"][test_indices]

        logger.info(f"Fold {i + 1} model training")

        # Train a LightGBM model for the current fold
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            df_fold_train[feature_columns],
            df_fold_train_target,
            eval_set=[(df_fold_valid[feature_columns], df_fold_valid_target)],
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=100),
                lgb.callback.log_evaluation(period=100),
            ],
        )

        models.append(lgb_model)

        # Evaluate model performance on the validation set
        fold_predictions = lgb_model.predict(df_fold_valid[feature_columns])
        fold_score = mean_absolute_error(fold_predictions, df_fold_valid_target)
        scores[f"MAE_fold_{i + 1}"] = fold_score
        logger.info(f"LGB fold: {i + 1}; MAE: {fold_score}")

        # Free up memory by deleting fold specific variables
        del df_fold_train, df_fold_train_target, df_fold_valid, df_fold_valid_target
        gc.collect()

    # Update the lgb_params with the average best iteration
    final_model_params = lgb_params.copy()

    final_model = lgb.LGBMRegressor(**final_model_params)
    final_model.fit(
        df_train_feats[feature_columns],
        df_train["target"],
        callbacks=[
            lgb.callback.log_evaluation(period=100),
        ],
    )
    # Append the final model to the list of models
    models.append(final_model)

    return models, scores


def tune_hp(
    df_train: pd.DataFrame,
    lgb_params_base: dict,
    lgb_params_tune: dict,
    pruner_params: dict,
    n_trials: int,
    split_day: int,
) -> optuna.study.Study:
    logger.info("Preparing data")
    df_train, df_valid = split(df_train, split_day)

    df_train_feats = generate_all_features(df_train, True)
    df_valid_feats = generate_all_features(df_valid, False)

    lgb_params = lgb_params_base

    def objective(trial):
        for param_name, param_settings in lgb_params_tune.items():
            if param_settings["type"] == "int":
                lgb_params[param_name] = trial.suggest_int(
                    param_name, **param_settings["kwargs"]
                )
            elif param_settings["type"] == "float":
                lgb_params[param_name] = trial.suggest_float(
                    param_name, **param_settings["kwargs"]
                )
            else:
                raise ValueError(
                    f"Unsupported hyperparameter type: {param_settings['type']}"
                )

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "l1")

        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            df_train_feats,
            df_train["target"],
            eval_set=[(df_valid_feats, df_valid["target"])],
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=100),
                lgb.callback.log_evaluation(period=100),
                pruning_callback,
            ],
        )
        predictions = lgb_model.predict(df_valid_feats)

        return mean_absolute_error(predictions, df_valid["target"])

    logger.info("Tuning hyperparameters")
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(**pruner_params),
        direction="minimize",
    )
    study.optimize(objective, n_trials=n_trials)

    return study
