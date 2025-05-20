import pandas as pd
from typing import Iterable

from ib_async_trading_bot.objects import DataTimeSeriesPandasDfObject

def get_dataset(
    data_obj: DataTimeSeriesPandasDfObject,
    df: pd.core.frame.DataFrame
) -> Iterable:
    assert data_obj.target in df.columns, f'{data_obj.target} not in Columns'
    if data_obj.columns is not None:
        for i in data_obj.columns:
            assert i in df.columns, f'{i} not in Columns'
    return df[data_obj.columns]