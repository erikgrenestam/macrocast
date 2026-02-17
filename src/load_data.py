"""Data loading, deduplication, and vintage management."""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from dnsql import DNSQL


def load_parquet(data_path: Path, deduplicate: bool = True) -> pd.DataFrame:
    """Load data from parquet, deduplicated by default.

    Args:
        data_path: Path to the parquet file.
        deduplicate: If True (default), deduplicate by keeping the latest
            valid_to for each (value_date, series). Set to False to return
            the raw data.
    """
    df = pd.read_parquet(data_path)
    if deduplicate:
        df = deduplicate_vintages(df)
    return df


def deduplicate_vintages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by keeping latest valid_to for each (value_date, series).

    For series without vintage data (valid_to is NaT), keep all records.
    """
    df = df.copy()
    df['valid_to_sort'] = df['valid_to'].fillna(pd.Timestamp('2200-12-31'))

    df = df.sort_values('valid_to_sort')
    df_dedup = df.drop_duplicates(
        subset=['value_date', 'internal_series_name'],
        keep='last'
    )

    return df_dedup.drop(columns=['valid_to_sort'])


def create_vintage(df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Create pseudo-vintage: data as it would have appeared on as_of_date.

    For series with vintages: use data where valid_from <= as_of_date < valid_to
    For series without vintages: use data where value_date < as_of_date
    """
    df = df.copy()

    has_vintage = df['valid_from'].notna()
    vintage_mask = (
        has_vintage &
        (df['valid_from'] <= as_of_date) &
        (df['valid_to'] > as_of_date)
    )

    no_vintage_mask = (~has_vintage) & (df['value_date'] < as_of_date)

    return df[vintage_mask | no_vintage_mask].copy()


def _get_tablename(test: bool = False) -> str:
    return 'MacroCastRawData_test' if test else 'MacroCastRawData'


def get_vintage(
    vintage_hash: Optional[str] = None,
    valid_ts: Optional[Union[pd.Timestamp, str]] = None,
    test: bool = False,
    deduplicate: bool = False,
    as_of_date: Optional[Union[pd.Timestamp, str]] = None,
) -> pd.DataFrame:
    """
    Retrieve a vintage from SQL.

    There are four scenarios:
    1. vintage_hash is None, valid_ts is None: returns the latest vintage
    2. vintage_hash is provided, valid_ts is None: returns all data for that hash
    3. vintage_hash is None, valid_ts is provided: returns the latest vintage
         valid at that timestamp
    4. Both provided: returns data for that specific vintage and timestamp

    Args:
        deduplicate: If True, deduplicate the result via deduplicate_vintages.
        as_of_date: If provided, filter to a pseudo-vintage as of this date
            via create_vintage.
    """
    t = _get_tablename(test)

    if isinstance(valid_ts, str):
        is_date_only = len(valid_ts.strip()) <= 10
        valid_ts = pd.to_datetime(valid_ts)
        if is_date_only:
            valid_ts = valid_ts.replace(hour=23, minute=59, second=59)

    if vintage_hash is None and valid_ts is None:
        logging.info("Fetching latest vintage")
        query = f"""
        SELECT *
        FROM area026.{t}
        WHERE vintage_ts = (
            SELECT MAX(vintage_ts)
            FROM area026.{t}
        );
        """
    elif vintage_hash is not None and valid_ts is None:
        logging.info(f"Fetching data for vintage hash: {vintage_hash}")
        query = f"""
        SELECT *
        FROM area026.{t}
        WHERE vintage_hash = '{vintage_hash}';
        """
    elif vintage_hash is None and valid_ts is not None:
        logging.info(f"Fetching latest vintage valid at timestamp: {valid_ts}")
        if valid_ts.microsecond != 0:
            logging.warning(f"valid_ts {valid_ts} has microseconds set. Rounding down to nearest second for SQL comparison.")
            valid_ts = valid_ts.replace(microsecond=0)
        query = f"""
        SELECT *
        FROM area026.{t}
        WHERE vintage_ts = (
            SELECT MAX(vintage_ts)
            FROM area026.{t}
            WHERE vintage_ts <= '{str(valid_ts)}'
        );
        """
    elif vintage_hash is not None and valid_ts is not None:
        logging.info(f"Fetching data for vintage hash: {vintage_hash} at timestamp: {valid_ts}")
        query = f"""
        SELECT *
        FROM area026.{t}
        WHERE vintage_hash = '{vintage_hash}' AND YEAR(vintage_ts) = {valid_ts.year} AND MONTH(vintage_ts) = {valid_ts.month}
        AND DAY(vintage_ts) = {valid_ts.day} AND DATEPART(HOUR, vintage_ts) = {valid_ts.hour}
        AND DATEPART(MINUTE, vintage_ts) = {valid_ts.minute} AND DATEPART(SECOND, vintage_ts) = {valid_ts.second};
        """
    else:
        raise ValueError("Invalid combination of vintage_hash and valid_ts parameters.")

    df = DNSQL.execute_query(query)
    if type(df) is not pd.DataFrame or df.empty:
        raise ValueError(f"No data found for vintage hash {vintage_hash} and timestamp {valid_ts}.")
    logging.info(f"Retrieved vintage data with shape {df.shape} and timestamp {df['vintage_ts'].iloc[0]}")

    if as_of_date is not None:
        if isinstance(as_of_date, str):
            as_of_date = pd.to_datetime(as_of_date)
        df = create_vintage(df, as_of_date)

    if deduplicate:
        df = deduplicate_vintages(df)

    return df


def compare_vintages(vintage_hash_1: str, vintage_hash_2: str, test: bool = False) -> pd.DataFrame:
    """Compares two vintages and returns a DataFrame highlighting the differences."""
    df1 = get_vintage(vintage_hash=vintage_hash_1, test=test)
    df2 = get_vintage(vintage_hash=vintage_hash_2, test=test)

    latest_ts_1 = df1['vintage_ts'].max()
    latest_ts_2 = df2['vintage_ts'].max()
    df1 = df1[df1['vintage_ts'] == latest_ts_1]
    df2 = df2[df2['vintage_ts'] == latest_ts_2]

    comparison_df = pd.merge(df1, df2, on=['value_date', 'internal_series_name'], suffixes=('_v1', '_v2'), how='outer', indicator=True)
    comparison_df['value_diff'] = comparison_df['value_v2'] - comparison_df['value_v1']
    comparison_df['is_different'] = comparison_df['value_diff'] != 0

    return comparison_df


def get_unique_vintages(test: bool = False) -> pd.DataFrame:
    """Returns a DataFrame with all unique vintage hashes and timestamps."""
    t = _get_tablename(test)
    query = f"""
    SELECT DISTINCT vintage_ts, vintage_hash
    FROM area026.{t};
    """

    df = DNSQL.execute_query(query)
    if type(df) is not pd.DataFrame or df.empty:
        raise ValueError("No vintages found in the database.")
    return df
