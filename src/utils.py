import numpy as np
import os
import pandas as pd

from loguru import logger
from typing import Sequence, Tuple


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce the memory usage of a Pandas DataFrame by downcasting numerical columns
    to more efficient data types while preserving data accuracy.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame whose memory usage needs to be optimized.
    verbose : bool, optional (default=True)
        Whether to print memory usage details before and after optimization.

    Returns:
    --------
    pd.DataFrame
        The optimized DataFrame with reduced memory footprint.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:

                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float32)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    if verbose:
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Decreased by {decrease:.2f}%")
    return df


class MockApi:
    """
    A mock API that simulates an environment for handling test data and submissions.

    This class mimics an iterative testing environment where test data is provided
    incrementally, and predictions must be submitted in response.

    Attributes:
    -----------
    input_paths : Sequence[str]
        A list of file paths pointing to the test dataset, revealed targets,
        and sample submission file.
    submission_id : int
        A unique identifier for the submission.
    group_id_column : str
        The column used for grouping data during iteration (default: "time_id").
    export_group_id_column : bool
        Whether to include the group ID column in exported data.
    _status : str
        The current status of the API, tracking progress through different stages.
    predictions : list[pd.DataFrame]
        A list storing user predictions for each iteration.
    """

    def __init__(self, test_root: str, submission_id: int):
        """
        Initializes the MockApi instance.

        Parameters:
        -----------
        test_root : str
            The root directory containing test-related data files.
        submission_id : int
            A unique identifier for the submission.
        """
        self.input_paths: Sequence[str] = [
            f"{test_root}/test.csv",
            f"{test_root}/revealed_targets.csv",
            f"{test_root}/sample_submission.csv",
        ]
        self.submission_id = submission_id
        self.group_id_column: str = "time_id"
        self.export_group_id_column: bool = False
        assert len(self.input_paths) >= 2

        self._status = "initialized"
        self.predictions = []

    def iter_test(self) -> Tuple[pd.DataFrame]:
        """
        Iterates over test data, yielding dataframes grouped by `group_id_column`.

        This method loads test data from the specified input paths and yields
        all rows belonging to the current `group_id_column` value in sequence.

        Yields:
        -------
        Tuple[pd.DataFrame]
            A tuple of DataFrames, each corresponding to a different dataset
            (e.g., test set, sample submission), filtered for the current group.
        """
        if self._status != "initialized":

            raise Exception(
                "WARNING: the real API can only iterate over `iter_test()` once."
            )

        dataframes = []
        for pth in self.input_paths:
            dataframes.append(pd.read_csv(pth, low_memory=False))
        group_order = dataframes[0][self.group_id_column].drop_duplicates().tolist()
        dataframes = [df.set_index(self.group_id_column) for df in dataframes]

        for group_id in group_order:
            self._status = "prediction_needed"
            current_data = []
            for df in dataframes:
                cur_df = df.loc[group_id].copy()
                # returning single line dataframes from df.loc requires special handling
                if not isinstance(cur_df, pd.DataFrame):
                    cur_df = pd.DataFrame(
                        {a: b for a, b in zip(cur_df.index.values, cur_df.values)},
                        index=[group_id],
                    )
                    cur_df.index.name = self.group_id_column
                cur_df = cur_df.reset_index(drop=not (self.export_group_id_column))
                current_data.append(cur_df)
            yield tuple(current_data)

            while self._status != "prediction_received":
                print(
                    """You must call `predict()` successfully before you can continue
                    with `iter_test()`""",
                    flush=True,
                )
                yield None

        save_dir = f"data/submissions/{self.submission_id}"
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/submission.csv", "w") as f_open:
            pd.concat(self.predictions).to_csv(f_open, index=False)
        self._status = "finished"

    def predict(self, user_predictions: pd.DataFrame):
        """
        Accepts user predictions and updates the API state.

        This method stores the provided predictions and allows `iter_test()`
        to continue to the next iteration.

        Parameters:
        -----------
        user_predictions : pd.DataFrame
            A DataFrame containing the user's predictions for the current test batch.

        Raises:
        -------
        Exception
            If predictions are made after the test set is completed, if predictions
            are submitted before calling `iter_test()`, or if the provided input
            is not a DataFrame.
        """
        if self._status == "finished":
            raise Exception("You have already made predictions for the full test set.")
        if self._status != "prediction_needed":
            raise Exception(
                "You must get the next test sample from `iter_test()` first."
            )
        if not isinstance(user_predictions, pd.DataFrame):
            raise Exception("You must provide a DataFrame.")

        self.predictions.append(user_predictions)
        self._status = "prediction_received"


def make_env(test_root: str, submission_id: int) -> MockApi:
    """
    Creates and returns a MockApi environment for handling test data and submissions.

    Parameters:
    -----------
    test_root : str
        The root directory containing the test data.
    submission_id : int
        The unique identifier for the submission.

    Returns:
    --------
    MockApi
        An instance of MockApi initialized with the provided test root and submission
        ID.
    """
    return MockApi(test_root, submission_id)
