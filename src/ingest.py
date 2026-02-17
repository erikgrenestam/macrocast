"""
Data ingestion module.

API fetchers: Pull raw series from DST, FRED, Eurostat, Stoxx, ECB.
Build pipeline: Aggregate all sources and upload vintages to SQL.
"""

import logging
from io import StringIO
from pathlib import Path
from typing import Optional

import eurostat
import numpy as np
import pandas as pd
import requests
import yaml
from dnsql import DNSQL

from macrocast import config, get_api_key, get_dataframe_short_hash
from load_data import get_vintage


# ---------------------------------------------------------------------------
# Ingest configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "ingest_config.yml") -> dict:
    """Loads the YAML configuration file."""
    try:
        script_dir = Path(__file__).parent
        full_path = script_dir / config_path

        with open(full_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {full_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

CONFIG = load_config()
DST_PARAMS = CONFIG['dst']
FRED_PARAMS = CONFIG['fred']
EUROSTAT_PARAMS = CONFIG['eurostat']
STOXX_PARAMS = CONFIG['stoxx']


# ---------------------------------------------------------------------------
# API fetchers
# ---------------------------------------------------------------------------

def _align_index_to_start_of_period(idx: pd.Index, freq: Optional[str] = None) -> pd.DatetimeIndex:
    """Aligns a DatetimeIndex to the start of its period based on inferred frequency."""
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("df does not have a datetime index")
    if freq is None:
        freq = pd.infer_freq(idx)
    if freq in ['QS', 'QE', 'QS-OCT']:
        return idx.to_period('Q').start_time
    elif freq in ['MS', 'ME']:
        return idx.to_period('M').start_time
    else:
        logging.warning(f"Frequency {freq} not recognized for alignment. Returning original df.")
        return idx


def get_dst_api_df(series_id: str) -> pd.DataFrame:
    """Fetches data from the DST API for a given series ID."""
    params = DST_PARAMS[series_id].copy()
    r = requests.post('https://api.statbank.dk/v1/data', json=params)
    r.raise_for_status()
    dst_df = pd.read_table(StringIO(r.text), sep=';')

    dst_df['TID'] = pd.to_datetime(dst_df['TID'], format='%YM%m')
    dst_df = dst_df.rename(columns={'TID': 'value_date', 'INDHOLD': 'value'})

    dst_df.loc[dst_df['value'] == '..', 'value'] = np.nan
    dst_df['value'] = pd.to_numeric(dst_df['value'], errors='coerce')
    dst_df['internal_series_name'] = params.get('name')
    dst_df['original_series_id'] = series_id
    dst_df['data_source'] = 'dst'
    dst_df['valid_from'] = None
    dst_df['valid_to'] = None

    dst_df = dst_df[['value_date', 'value', 'original_series_id', 'internal_series_name', 'data_source', 'valid_from', 'valid_to']]
    dst_df = dst_df.set_index('value_date').sort_index()
    return dst_df


def get_fred_api_df(series_id: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetches data from the FRED API for a given series ID."""
    if api_key is None:
        api_key = get_api_key('fred')

    params = FRED_PARAMS[series_id].copy()

    r = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&realtime_start={config.fred_realtime_start}&realtime_end=9999-12-31&api_key={api_key}&file_type=json")
    r.raise_for_status()
    fred_df = pd.DataFrame(r.json()['observations'])

    fred_df['valid_from'] = pd.to_datetime(fred_df['realtime_start'])
    valid_to_replaced = fred_df['realtime_end'].astype(str).str.replace(r"^9999-12-31", config.fred_realtime_end, regex=True)
    fred_df['valid_to'] = pd.to_datetime(valid_to_replaced)

    fred_df['value_date'] = pd.to_datetime(fred_df['date'])
    fred_df['value'] = pd.to_numeric(fred_df['value'], errors='coerce')
    fred_df['internal_series_name'] = params.get('name')
    fred_df['original_series_id'] = series_id
    fred_df['data_source'] = 'fred'

    fred_df = fred_df[['value_date', 'value', 'original_series_id', 'internal_series_name', 'data_source', 'valid_from', 'valid_to']]
    fred_df = fred_df.set_index('value_date').sort_index()
    return fred_df


def get_stoxx_df(series_id: str) -> pd.DataFrame:
    """Fetches data from the Stoxx website for a given series ID."""
    params = STOXX_PARAMS[series_id].copy()

    r = requests.get(f'https://www.stoxx.com/document/Indices/Current/HistoricalData/h_{series_id}.txt')
    r.raise_for_status()
    stoxx_df = pd.read_table(StringIO(r.text), sep=';')

    stoxx_df['value_date'] = pd.to_datetime(stoxx_df['Date'], format='%d.%m.%Y')
    stoxx_df['value'] = pd.to_numeric(stoxx_df['Indexvalue'], errors='coerce')
    stoxx_df['internal_series_name'] = params.get('name')
    stoxx_df['original_series_id'] = series_id
    stoxx_df['data_source'] = 'stoxx'
    stoxx_df['valid_from'] = None
    stoxx_df['valid_to'] = None

    stoxx_df = stoxx_df[['value_date', 'value', 'original_series_id', 'internal_series_name', 'data_source', 'valid_from', 'valid_to']]
    stoxx_df = stoxx_df.set_index('value_date').sort_index()
    return stoxx_df


def get_ecb_api_df(dataset: str) -> pd.DataFrame:
    # TODO: standardize date parsing, index setting, etc.
    return pd.read_csv(f"https://data-api.ecb.europa.eu/service/data/{dataset}?format=csvdata", sep=',')


def get_eurostat_api_df(series_id: str) -> pd.DataFrame:
    """Fetches data from the Eurostat API for a given series ID."""
    params = EUROSTAT_PARAMS[series_id].copy()

    series_code = params.pop('datacode')
    pars = eurostat.get_pars(series_code)
    eurostat_df = eurostat.get_data_df(series_code, filter_pars=params)
    if eurostat_df is None or eurostat_df.empty:
        raise ValueError(f"No data returned from Eurostat for series {series_code} with params {params}")

    series_freq = eurostat_df['freq'].iloc[0]
    if series_freq == 'Q':
        raise NotImplementedError("Quarterly Eurostat data handling not yet implemented.")

    eurostat_df = eurostat_df.dropna(axis='columns')
    cols_to_drop = [col for col in eurostat_df.columns if col.startswith(tuple(pars))]
    logging.debug(f"Dropping Eurostat columns: {cols_to_drop}")
    eurostat_df = eurostat_df.drop(columns=cols_to_drop)
    eurostat_df = pd.melt(eurostat_df, value_name='value')

    eurostat_df['internal_series_name'] = params.get('name')
    eurostat_df['original_series_id'] = series_id
    eurostat_df['data_source'] = 'eurostat'
    eurostat_df['valid_from'] = None
    eurostat_df['valid_to'] = None
    eurostat_df['value_date'] = pd.to_datetime(eurostat_df['variable'], format='%Y-%m')

    eurostat_df = eurostat_df[['value_date', 'value', 'original_series_id', 'internal_series_name', 'data_source', 'valid_from', 'valid_to']]
    eurostat_df = eurostat_df.set_index('value_date').sort_index()
    return eurostat_df


# ---------------------------------------------------------------------------
# Build combined dataset from all sources
# ---------------------------------------------------------------------------

def build_dst_data() -> pd.DataFrame:
    """Builds a combined DataFrame from all DST series defined in the configuration."""
    dfs = []
    logging.info("Building DST data")
    for table in DST_PARAMS.keys():
        logging.info(f"Fetching DST table {table}")
        dfs.append(get_dst_api_df(table))
    return pd.concat(dfs, axis=0)


def build_fred_data() -> pd.DataFrame:
    """Builds a combined DataFrame from all FRED series defined in the configuration."""
    dfs = []
    logging.info("Building FRED data")
    for series_id in FRED_PARAMS.keys():
        logging.info(f"Fetching FRED series {series_id}")
        dfs.append(get_fred_api_df(series_id=series_id))
    return pd.concat(dfs, axis=0)


def build_eurostat_data() -> pd.DataFrame:
    """Builds a combined DataFrame from all Eurostat series defined in the configuration."""
    dfs = []
    logging.info("Building Eurostat data")
    for series_id in EUROSTAT_PARAMS.keys():
        logging.info(f"Fetching Eurostat series {series_id}")
        dfs.append(get_eurostat_api_df(series_id=series_id))
    return pd.concat(dfs, axis=0)


def build_stoxx_data() -> pd.DataFrame:
    """Builds a combined DataFrame from all Stoxx series defined in the configuration."""
    dfs = []
    logging.info("Building Stoxx data")
    for series_id in STOXX_PARAMS.keys():
        logging.info(f"Fetching Stoxx series {series_id}")
        dfs.append(get_stoxx_df(series_id=series_id))
    return pd.concat(dfs, axis=0)


def build_data(upload_vintage: bool = True, test: bool = False) -> pd.DataFrame:
    """Aggregates data from all sources (DST, FRED, Eurostat, Stoxx) into a single DataFrame."""
    logging.info("Building combined dataset from all sources...")

    dfs = [
        build_dst_data(),
        build_fred_data(),
        build_eurostat_data(),
        build_stoxx_data()
    ]

    dfs = [df for df in dfs if not df.empty]

    if not dfs:
        raise ValueError("No data available from any source.")

    final_df = pd.concat(dfs, axis=0)
    final_df = final_df.sort_index()
    logging.info(f"Combined build complete. Final shape: {final_df.shape}")

    if upload_vintage:
        logging.info("Uploading data to datahub...")
        ingest_data_vintage(df=final_df, test=test)

    return final_df


# ---------------------------------------------------------------------------
# Vintage upload
# ---------------------------------------------------------------------------

def ingest_data_vintage(df: pd.DataFrame, test: bool) -> None:
    short_hash = get_dataframe_short_hash(df)
    logging.info(f"Generated vintage hash: {short_hash}")

    df_out = df.copy()
    df_out['vintage_ts'] = pd.Timestamp.now()
    df_out['vintage_hash'] = short_hash
    df_out = df_out.reset_index(names='value_date')
    logging.debug(f"Prepared DataFrame for ingestion with dtype {df_out.dtypes.to_dict()} and shape {df_out.shape}")

    tablename = 'MacroCastRawData_test' if test else 'MacroCastRawData'
    DNSQL.df_to_sql(df=df_out,
                    tablename=tablename,
                    if_exists='append',
                    schema='area026',
                    index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = get_vintage(test=True)
    df.drop(columns=['vintage_ts', 'vintage_hash'], inplace=True)
    df.to_parquet('example_data.parquet', index=False)
