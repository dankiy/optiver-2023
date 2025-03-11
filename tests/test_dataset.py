import pandas as pd
import pytest

from src.dataset import split


@pytest.fixture
def sample_data():
    return {
        "date_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }


@pytest.mark.parametrize("split_day", [3, 5, 7, 10, 0])
def test_split(split_day, sample_data):
    df = pd.DataFrame(sample_data)

    split_day = 5

    df_train, df_valid = split(df, split_day)

    assert df_train["date_id"].max() <= split_day
    assert df_valid["date_id"].min() > split_day
