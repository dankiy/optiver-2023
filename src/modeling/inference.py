import numpy as np
import pandas as pd

from loguru import logger
from sklearn.base import RegressorMixin

from ..features import generate_all_features
from ..utils import make_env


def weighted_average(models: list[RegressorMixin]) -> list[int]:
    """
    Computes weights for a list of models using an exponential decay scheme.

    The first model receives the highest weight, and subsequent models receive
    progressively smaller weights.

    Parameters:
    -----------
    models : list[RegressorMixin]
        A list of regression models for which weights are to be computed.

    Returns:
    --------
    list[int]
        A list of weights corresponding to the models, computed using an
        exponential decay formula.
    """
    weights = []
    n = len(models)
    for idx in range(1, n + 1):
        k = 2 if idx == 1 else idx
        weights.append(1 / (2 ** (n + 1 - k)))
    return weights


def generate_submission(
    models: list[RegressorMixin],
    test_root: str,
    submission_id: int,
    clip_min: int,
    clip_max: int,
    tail_size: int,
):
    """
    Generates a submission by iteratively predicting targets using an ensemble
    of regression models and writing the predictions to the competition environment.

    This function processes test data in batches, applies feature engineering,
    generates predictions using a weighted average of multiple models, clips the
    predictions to a specified range, and submits the results using mock environment.

    Parameters:
    -----------
    models : list[RegressorMixin]
        A list of trained regression models used for making predictions.
    test_root : str
        The root directory containing the test dataset.
    submission_id : int
        A unique identifier for the submission.
    clip_min : int
        The minimum value to which predictions are clipped.
    clip_max : int
        The maximum value to which predictions are clipped.
    tail_size : int
        The number of most recent data points to retain for feature engineering.
    """
    env = make_env(test_root, submission_id)
    iter_test = env.iter_test()
    counter = 0
    y_min, y_max = clip_min, clip_max
    predictions = []
    cache = pd.DataFrame()

    # Weights for each fold model
    lgb_model_weights = weighted_average(models)

    for test, revealed_targets, sample_prediction in iter_test:
        cache = pd.concat([cache, test], ignore_index=True, axis=0)
        if counter > 0:
            cache = (
                cache.groupby(["stock_id"])
                .tail(tail_size)
                .sort_values(by=["date_id", "seconds_in_bucket", "stock_id"])
                .reset_index(drop=True)
            )
        feat = generate_all_features(cache, False)[-len(test) :].drop(
            columns="currently_scored"
        )
        feature_columns = feat.columns
        logger.info(f"Features shape is: {feat.shape}")

        # Generate predictions for each model and calculate the weighted average
        lgb_predictions = np.zeros(len(test))
        for model, weight in zip(models, lgb_model_weights):
            lgb_predictions += weight * model.predict(feat[feature_columns])

        predictions = lgb_predictions

        # Using mean predictions rather than zero sum
        final_predictions = predictions - np.mean(predictions)
        clipped_predictions = np.clip(final_predictions, y_min, y_max)
        sample_prediction["target"] = clipped_predictions
        env.predict(sample_prediction)
        counter += 1
