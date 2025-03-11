import opendatasets as od
import pandas as pd


def download(data_url: str):
    """
    Download competition data using the input URL.

    Parameters:
    -----------
    data_url : str
        The URL from which the competition data will be downloaded.
    """
    od.download(data_url, "data/")


def split(df: pd.DataFrame, split_day: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits a DataFrame into training and validation sets based on a given day.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing a "date_id" column used for splitting.
    split_day : int
        The threshold day used to divide the dataset into training and validation sets.
        Rows with "date_id" less than or equal to this value are assigned to the
        training set, while rows with "date_id" greater than this value are assigned to
        the validation set.

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames:
        - Training set.
        - Validation set.
    """
    df_train = df[df["date_id"] <= split_day]
    df_valid = df[df["date_id"] > split_day]
    return df_train, df_valid
