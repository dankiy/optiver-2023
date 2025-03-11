import gc
import joblib as jb
import json
import numpy as np
import os
import pandas as pd
import polars as pl

from itertools import combinations
from numba import njit, prange
from typing import Optional


@njit(parallel=True)
def triplet_imbalance_numba(
    df_values: np.ndarray, comb_indices: list[tuple[int, int, int]]
) -> np.ndarray:
    """
    Computes triplet imbalance features for given column indices.

    Parameters:
    -----------
    df_values : np.ndarray
        A 2D NumPy array containing the values of the DataFrame columns.
    comb_indices : list[tuple[int, int, int]]
        A list of tuples, where each tuple represents the indices of three columns
        used to compute the imbalance feature.

    Returns:
    --------
    np.ndarray
        A 2D NumPy array where each column contains the computed triplet imbalance
        values for the corresponding column triplet.
    """
    num_rows, num_combinations = df_values.shape[0], len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val, min_val = max(
                df_values[j, a], df_values[j, b], df_values[j, c]
            ), min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = (
                df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            )

            imbalance_features[j, i] = (
                np.nan
                if mid_val == min_val
                else (max_val - mid_val) / (mid_val - min_val)
            )

    return imbalance_features


def compute_triplet_features(price: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes imbalance features for triplets of price-related columns.

    Parameters:
    -----------
    price : list[str]
        A list of column names representing different price-related features.
    df : pd.DataFrame
        The input DataFrame containing the price-related columns.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the computed triplet imbalance features, where each
        column corresponds to an imbalance metric computed for a specific triplet of
        price columns.
    """
    df_values = df[price].values
    comb_indices = [
        (price.index(a), price.index(b), price.index(c))
        for a, b, c in combinations(price, 3)
    ]
    features_array = triplet_imbalance_numba(df_values, comb_indices)
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)
    return features


def add_agg_features(
    df: pd.DataFrame, prices: list[str], sizes: list[str]
) -> pd.DataFrame:
    """
    Adds aggregated statistical features (mean, standard deviation, skewness, and
    kurtosis) for price and size-related columns. Also computes lagged and return
    features for certain columns over different time windows.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing market data.
    prices : list[str]
        A list of column names corresponding to price-related features.
    sizes : list[str]
        A list of column names corresponding to size-related features.

    Returns:
    --------
    pd.DataFrame
        The modified DataFrame with added aggregated and lagged/return features.
    """
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)

    for col in [
        "matched_size",
        "imbalance_size",
        "reference_price",
        "imbalance_buy_sell_flag",
    ]:
        for window in [1, 3, 5, 10]:
            df[f"{col}_shift_{window}"] = df.groupby("stock_id")[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby("stock_id")[col].pct_change(window)
    return df


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling difference-based features for price and size-related columns over
    different time windows. Also calculates derived features for price and size changes.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing stock market data.

    Returns:
    --------
    pd.DataFrame
        The modified DataFrame with additional difference and rolling statistical
        features.
    """
    for col in [
        "ask_price",
        "bid_price",
        "ask_size",
        "bid_size",
        "weighted_wap",
        "price_spread",
    ]:
        for window in [1, 3, 5, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)

    # V4 feature
    for window in [3, 5, 10]:
        df[f"price_change_diff_{window}"] = (
            df[f"bid_price_diff_{window}"] - df[f"ask_price_diff_{window}"]
        )
        df[f"size_change_diff_{window}"] = (
            df[f"bid_size_diff_{window}"] - df[f"ask_size_diff_{window}"]
        )

    # V5 - rolling diff
    # Convert from pandas to Polars
    pl_df = pl.from_pandas(df)

    # Define the windows and columns for which you want to calculate the rolling
    # statistics
    windows = [3, 5, 10]
    columns = ["ask_price", "bid_price", "ask_size", "bid_size"]

    # prepare the operations for each column and window
    group = ["stock_id"]
    expressions = []

    # Loop over each window and column to create the rolling mean and std expressions
    for window in windows:
        for col in columns:
            rolling_mean_expr = (
                pl.col(f"{col}_diff_{window}")
                .rolling_mean(window)
                .over(group)
                .alias(f"rolling_diff_{col}_{window}")
            )

            rolling_std_expr = (
                pl.col(f"{col}_diff_{window}")
                .rolling_std(window)
                .over(group)
                .alias(f"rolling_std_diff_{col}_{window}")
            )

            expressions.append(rolling_mean_expr)
            expressions.append(rolling_std_expr)

    # Run the operations using Polars' lazy API
    lazy_df = pl_df.lazy().with_columns(expressions)

    # Execute the lazy expressions and overwrite the pl_df variable
    pl_df = lazy_df.collect()

    # Convert back to pandas if necessary
    df = pl_df.to_pandas()
    gc.collect()

    return df


def add_imbalance_features(
    df: pd.DataFrame, stock_weights: dict[str, float]
) -> pd.DataFrame:
    """
    Adds imbalance-related features derived from price and size columns.
    Computes additional statistical and liquidity-based metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing stock market data.
    stock_weights : dict[str, float]
        A dictionary mapping stock IDs to their respective weights.

    Returns:
    --------
    pd.DataFrame
        The modified DataFrame with additional imbalance features.
    """
    # Define lists of price and size-related column names
    prices = [
        "reference_price",
        "far_price",
        "near_price",
        "ask_price",
        "bid_price",
        "wap",
    ]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval(
        "(imbalance_size-matched_size)/(matched_size+imbalance_size)"
    )
    df["size_imbalance"] = df.eval("bid_size / ask_size")

    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    for c in [["ask_price", "bid_price", "wap", "reference_price"], sizes]:
        triplet_feature = compute_triplet_features(c, df)
        df[triplet_feature.columns] = triplet_feature.values

    df["stock_weights"] = df["stock_id"].map(stock_weights)
    df["weighted_wap"] = df["stock_weights"] * df["wap"]
    df["wap_momentum"] = df.groupby("stock_id")["weighted_wap"].pct_change(periods=6)

    df["imbalance_momentum"] = (
        df.groupby(["stock_id"])["imbalance_size"].diff(periods=1) / df["matched_size"]
    )
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(["stock_id"])["price_spread"].diff()
    df["price_pressure"] = df["imbalance_size"] * (df["ask_price"] - df["bid_price"])
    df["market_urgency"] = df["price_spread"] * df["liquidity_imbalance"]
    df["depth_pressure"] = (df["ask_size"] - df["bid_size"]) * (
        df["far_price"] - df["near_price"]
    )

    df["spread_depth_ratio"] = (df["ask_price"] - df["bid_price"]) / (
        df["bid_size"] + df["ask_size"]
    )
    df["mid_price_movement"] = (
        df["mid_price"]
        .diff(periods=5)
        .apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    )

    df["micro_price"] = (
        (df["bid_price"] * df["ask_size"]) + (df["ask_price"] * df["bid_size"])
    ) / (df["bid_size"] + df["ask_size"])
    df["relative_spread"] = (df["ask_price"] - df["bid_price"]) / df["wap"]

    # Calculate various statistical aggregation features
    df = add_agg_features(df, prices, sizes)

    # Calculate diff features for specific columns
    df = add_diff_features(df)

    df["mid_price*volume"] = df["mid_price_movement"] * df["volume"]
    df["harmonic_imbalance"] = df.eval("2 / ((1 / bid_size) + (1 / ask_size))")

    return df


def fill_inf_values(df) -> pd.DataFrame:
    """
    Replaces infinite values in a DataFrame with zero.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing numeric values, potentially with infinite values.

    Returns:
    --------
    pd.DataFrame
        The modified DataFrame with infinite values replaced by zero.
    """
    for col in df.columns:
        df[col] = df[col].replace([np.inf, -np.inf], 0)
    return df


def get_global_features(df_train: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Compute global stock-level features based on bid, ask sizes, and prices.

    Parameters:
    -----------
    df_train : pd.DataFrame
        The training dataset containing stock market data.

    Returns:
    --------
    dict[str, pd.Series]
        A dictionary containing stock-level aggregated statistics such as median,
        standard deviation, and peak-to-peak (max-min) values for sizes and prices.
    """
    grouped = df_train.groupby("stock_id")
    return {
        "median_size": grouped["bid_size"].median() + grouped["ask_size"].median(),
        "std_size": grouped["bid_size"].std() + grouped["ask_size"].std(),
        "ptp_size": grouped["bid_size"].max() - grouped["bid_size"].min(),
        "median_price": grouped["bid_price"].median() + grouped["ask_price"].median(),
        "std_price": grouped["bid_price"].std() + grouped["ask_price"].std(),
        "ptp_price": grouped["bid_price"].max() - grouped["ask_price"].min(),
    }


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-related features such as day of the week, minute, and time to market close.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing market timestamps.

    Returns:
    --------
    pd.DataFrame
        The dataset with additional temporal features.
    """
    df["dow"] = df["date_id"] % 5  # Day of the week (assuming a 5-day market week)
    df["seconds"] = df["seconds_in_bucket"] % 60
    df["minute"] = df["seconds_in_bucket"] // 60
    df["time_to_market_close"] = 540 - df["seconds_in_bucket"]
    return df


def add_global_stock_features(
    df: pd.DataFrame, global_stock_id_feats: dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Map global stock-level features to individual stocks in the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing stock IDs.

    global_stock_id_feats : dict[str, pd.Series]
        A dictionary of precomputed global stock-level features.

    Returns:
    --------
    pd.DataFrame
        The dataset with added global stock features.
    """
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())
    return df


def add_stock_features(
    df: pd.DataFrame, global_stock_id_feats: dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Compute and add stock-related features including temporal and global stock-level
    statistics.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing stock market data.

    global_stock_id_feats : dict[str, pd.Series]
        A dictionary of global stock-level aggregated statistics.

    Returns:
    --------
    pd.DataFrame
        The dataset enriched with stock-related features.
    """
    df = add_temporal_features(df)
    df = add_global_stock_features(df, global_stock_id_feats)
    return df


def generate_all_features(
    df: pd.DataFrame, train: bool
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Generate all required features for training and validation datasets.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataset containing stock market data.

    Returns:
    --------
    tuple[pd.DataFrame, Optional[pd.DataFrame]]
        The processed training dataset and optionally the validation dataset with added
        features.
    """
    cols = [c for c in df.columns if c not in {"row_id", "time_id", "target"}]

    # Load stock weights from configuration
    with open("configs/stock_weights.json") as f:
        stock_weights = json.load(f)["weights"]
        stock_weights = {int(k): v for k, v in enumerate(stock_weights)}

    df = df[cols]
    global_feats_path = "data/interim/global_features.jb"
    if train:
        global_stock_id_feats = get_global_features(df)
        global_feats_folder = "/".join(global_feats_path.split("/")[:-1])
        os.makedirs(global_feats_folder, exist_ok=True)
        jb.dump(global_stock_id_feats, global_feats_path)
    else:
        global_stock_id_feats = jb.load(global_feats_path)

    # Compute imbalance and stock features
    df = add_imbalance_features(df, stock_weights)
    df = add_stock_features(df, global_stock_id_feats)
    df = fill_inf_values(df)

    gc.collect()

    feature_cols = [
        c for c in df.columns if c not in {"row_id", "target", "time_id", "date_id"}
    ]

    return df[feature_cols]
