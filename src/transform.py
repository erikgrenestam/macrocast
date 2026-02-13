"""Data transformation pipeline for BART forecasting."""

import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

from config import ModelConfig, VariableSpec


@dataclass
class TransformedData:
    """Container for transformed data ready for modeling."""

    X: np.ndarray  # Shape: (n_samples, n_features)
    y: np.ndarray  # Shape: (n_samples,)
    feature_names: list[str]
    dates: pd.DatetimeIndex
    metadata: dict
    target_original: Optional[pd.Series] = None  # Pre-transformation target series
    transformer: Optional['Transformer'] = None  # For inverse_standardize


class DataLoader:
    """Load and deduplicate raw parquet data."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self._raw_data: Optional[pd.DataFrame] = None

    def load(self, deduplicate: bool = True) -> pd.DataFrame:
        """Load data from parquet, deduplicated by default.

        Args:
            deduplicate: If True (default), deduplicate by keeping the latest
                valid_to for each (value_date, series). Set to False to return
                the raw data.
        """
        df = pd.read_parquet(self.data_path)
        if deduplicate:
            df = self.deduplicate(df)
        return df

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate by keeping latest valid_to for each (value_date, series).

        For series without vintage data (valid_to is NaT), keep all records.
        """
        df = df.copy()
        df['valid_to_sort'] = df['valid_to'].fillna(pd.Timestamp('2200-12-31'))

        # Sort and keep last (latest vintage)
        df = df.sort_values('valid_to_sort')
        df_dedup = df.drop_duplicates(
            subset=['value_date', 'internal_series_name'],
            keep='last'
        )

        return df_dedup.drop(columns=['valid_to_sort'])

    def create_vintage(
        self,
        df: pd.DataFrame,
        as_of_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Create pseudo-vintage: data as it would have appeared on as_of_date.

        For series with vintages: use data where valid_from <= as_of_date < valid_to
        For series without vintages: use data where value_date < as_of_date
        """
        df = df.copy()

        # Series with vintage tracking
        has_vintage = df['valid_from'].notna()
        vintage_mask = (
            has_vintage &
            (df['valid_from'] <= as_of_date) &
            (df['valid_to'] > as_of_date)
        )

        # Series without vintage tracking (use simple date filter)
        no_vintage_mask = (~has_vintage) & (df['value_date'] < as_of_date)

        return df[vintage_mask | no_vintage_mask].copy()


class FrequencyInferrer:
    """Infer and handle different data frequencies."""

    @staticmethod
    def infer_frequency(dates: pd.Series) -> str:
        """
        Infer frequency from date series.

        Returns: 'D' (daily), 'M' (monthly), 'Q' (quarterly), 'A' (annual)
        """
        dates_sorted = dates.sort_values()
        if len(dates_sorted) < 2:
            return 'M'  # Default to monthly

        median_gap = dates_sorted.diff().median().days

        if median_gap <= 7:
            return 'D'
        elif 20 <= median_gap <= 35:
            return 'M'
        elif 80 <= median_gap <= 100:
            return 'Q'
        else:
            return 'A'

    @staticmethod
    def aggregate_to_monthly(
        df: pd.DataFrame,
        method: str = 'last'
    ) -> pd.DataFrame:
        """
        Aggregate higher-frequency data to monthly.

        Args:
            df: DataFrame with 'value_date' and 'value' columns
            method: 'mean', 'sd', 'first', 'last'
        """
        df = df.copy()
        df['year_month'] = df['value_date'].dt.to_period('M')

        if method == 'mean':
            result = df.groupby('year_month')['value'].mean()
        elif method == 'sd':
            result = df.groupby('year_month')['value'].std()
        elif method == 'first':
            result = df.groupby('year_month')['value'].first()
        elif method == 'last':
            result = df.groupby('year_month')['value'].last()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Convert period back to timestamp (first day of month)
        result.index = result.index.to_timestamp()
        return result.to_frame('value')


class Transformer:
    """Apply transformations to time series."""

    def __init__(self, standardize: bool = True):
        self.standardize = standardize
        self._std_params: dict[str, dict[str, float]] = {}

    def apply_transformation(
        self,
        series: pd.Series,
        transformation: str,
        series_name: Optional[str] = None,
    ) -> pd.Series:
        """
        Apply specified transformation, then optionally standardize to mean 0, std 1.

        Args:
            series: Input time series.
            transformation: One of 'none', 'log', 'diff', '12m_diff', 'log_diff', 'log_12m_diff'.
            series_name: Key used to store standardization params for later
                inversion. If None, standardization params are stored under the
                transformation name.

        Returns: Transformed (and possibly standardized) series.
        """
        if transformation == 'none':
            result = series
        elif transformation == 'log':
            result = np.log(series)
        elif transformation == 'diff':
            result = series.diff()
        elif transformation == '12m_diff':
            result = series.diff(12)
        elif transformation == 'log_diff':
            result = np.log(series).diff()
        elif transformation == 'log_12m_diff':
            result = np.log(series).diff(12)
        else:
            raise ValueError(f"Unknown transformation: {transformation}")

        if self.standardize:
            key = series_name if series_name is not None else transformation
            mean = result.mean()
            std = result.std()
            if std == 0 or np.isnan(std):
                std = 1.0
            self._std_params[key] = {'mean': mean, 'std': std}
            result = (result - mean) / std

        return result

    def inverse_standardize(
        self,
        series: pd.Series,
        series_name: Optional[str] = None,
        transformation: Optional[str] = None,
    ) -> pd.Series:
        """
        Reverse standardization applied during apply_transformation.

        Args:
            series: Standardized series.
            series_name: Key used when apply_transformation was called.
            transformation: Fallback key (used when series_name is None).

        Returns: Series in the pre-standardized (but still transformed) scale.
        """
        key = series_name if series_name is not None else transformation
        if key is None or key not in self._std_params:
            return series
        params = self._std_params[key]
        return series * params['std'] + params['mean']

    @staticmethod
    def inverse_transformation(
        series: pd.Series,
        transformation: str,
        original_series: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Inverse transform for forecast evaluation.

        For differencing, requires original_series for cumulative sum.
        """
        if transformation == 'none':
            return series
        elif transformation == 'log':
            return np.exp(series)
        elif transformation == 'diff':
            if original_series is None:
                raise ValueError("Need original series for diff inversion")
            return series.cumsum() + original_series.iloc[0]
        elif transformation == '12m_diff':
            if original_series is None:
                raise ValueError("Need original series for 12m_diff inversion")
            # Reconstruct level
            result = series.copy()
            for i in range(len(result)):
                if i < 12:
                    result.iloc[i] = original_series.iloc[i] + series.iloc[i]
                else:
                    result.iloc[i] = result.iloc[i-12] + series.iloc[i]
            return result
        elif transformation == 'log_diff':
            if original_series is None:
                raise ValueError("Need original series for log_diff inversion")
            log_level = series.cumsum() + np.log(original_series.iloc[0])
            return np.exp(log_level)
        elif transformation == 'log_12m_diff':
            if original_series is None:
                raise ValueError("Need original series for log_12m_diff inversion")
            log_original = np.log(original_series)
            log_result = series.copy()
            for i in range(len(log_result)):
                if i < 12:
                    log_result.iloc[i] = log_original.iloc[i] + series.iloc[i]
                else:
                    log_result.iloc[i] = log_result.iloc[i-12] + series.iloc[i]
            return np.exp(log_result)
        else:
            raise ValueError(f"Unknown transformation: {transformation}")


class LagFeatureBuilder:
    """Build lag features respecting publication lags and forecast horizons."""

    @staticmethod
    def create_lag_features(
        series: pd.Series,
        n_lags: int,
        horizon: int,
        publication_lag: int,
        feature_name: str
    ) -> pd.DataFrame:
        """
        Create lag features for direct forecasting.

        For direct h-step ahead forecasting:
        - First available lag is L_{h + publication_lag}
        - Create n_lags consecutive lags from there

        Example: h=6, publication_lag=1, n_lags=3
        - Creates: L7, L8, L9 (i.e., t-7, t-8, t-9)

        Args:
            series: Time series data (index must be datetime)
            n_lags: Number of lags to create
            horizon: Forecast horizon (1 = 1-month ahead)
            publication_lag: Publication delay in months
            feature_name: Base name for features

        Returns: DataFrame with lag columns
        """
        first_lag = horizon + publication_lag

        lag_data = {}
        for i in range(n_lags):
            lag_num = first_lag + i
            lag_series = series.shift(lag_num)
            lag_data[f"{feature_name}_L{lag_num}"] = lag_series

        return pd.DataFrame(lag_data, index=series.index)


class DataTransformer:
    """Main transformer orchestrating the full pipeline."""

    def __init__(self, config: ModelConfig, data_path: Path):
        self.config = config
        self.loader = DataLoader(data_path)
        self.freq_inferrer = FrequencyInferrer()
        self.transformer = Transformer()

    def prepare_data(
        self,
        horizon: int,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> TransformedData:
        """
        Prepare complete dataset for a specific forecast horizon.

        Args:
            horizon: Forecast horizon (e.g., 6 for 6-month ahead)
            as_of_date: If provided, create pseudo-vintage as of this date

        Returns: TransformedData with X, y arrays ready for BART
        """
        # 1. Load data
        if as_of_date:
            raw_df = self.loader.load(deduplicate=False)
            df = self.loader.create_vintage(raw_df, as_of_date)
        else:
            df = self.loader.load()

        # 2. Prepare target variable (keep original before transformation)
        target_original = self._extract_target_series(df)
        target_series = self._prepare_target(df)

        # 3. Prepare feature variables
        feature_df = self._prepare_features(df, horizon)

        # 4. Align target with features (handle forecast horizon)
        y_aligned = target_series.shift(-horizon)

        # 5. Merge and remove NaNs
        full_df = feature_df.copy()
        full_df['target'] = y_aligned

        full_df = full_df.dropna()

        # 6. Extract arrays
        X = full_df.drop(columns=['target']).values
        y = full_df['target'].values
        feature_names = full_df.drop(columns=['target']).columns.tolist()
        dates = full_df.index

        return TransformedData(
            X=X,
            y=y,
            feature_names=feature_names,
            dates=dates,
            metadata={
                'horizon': horizon,
                'as_of_date': as_of_date,
                'n_samples': len(X),
                'n_features': X.shape[1]
            },
            target_original=target_original,
            transformer=self.transformer
        )

    def _extract_target_series(self, df: pd.DataFrame) -> pd.Series:
        """Extract target series at monthly frequency, before any transformation."""
        target_spec = self.config.target
        target_df = df[
            df['internal_series_name'] == target_spec.internal_series_name
        ].copy()

        if len(target_df) == 0:
            raise ValueError(f"Target series '{target_spec.internal_series_name}' not found in data")

        freq = self.freq_inferrer.infer_frequency(target_df['value_date'])
        if freq != 'M':
            target_df = self.freq_inferrer.aggregate_to_monthly(target_df, method='last')
            return target_df['value'].sort_index()
        else:
            return target_df.set_index('value_date')['value'].sort_index()

    def _prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Prepare target variable with transformations."""
        target_spec = self.config.target

        # Filter to target series
        target_df = df[
            df['internal_series_name'] == target_spec.internal_series_name
        ].copy()

        if len(target_df) == 0:
            raise ValueError(f"Target series '{target_spec.internal_series_name}' not found in data")

        # Infer frequency and aggregate to monthly
        freq = self.freq_inferrer.infer_frequency(target_df['value_date'])
        if freq != 'M':
            target_df = self.freq_inferrer.aggregate_to_monthly(
                target_df, method='last'
            )
            # After aggregation, date is already the index
            target_series = target_df['value'].sort_index()
        else:
            target_series = target_df.set_index('value_date')['value'].sort_index()

        # Apply transformation
        target_series = self.transformer.apply_transformation(
            target_series, target_spec.transformation,
            series_name=target_spec.internal_series_name
        )

        return target_series

    def _prepare_features(
        self,
        df: pd.DataFrame,
        horizon: int
    ) -> pd.DataFrame:
        """Prepare all feature variables with lags."""
        all_features = []

        for feat_spec in self.config.features:
            # Filter to this series
            feat_df = df[
                df['internal_series_name'] == feat_spec.internal_series_name
            ].copy()

            if len(feat_df) == 0:
                print(f"Warning: Feature series '{feat_spec.internal_series_name}' not found in data")
                continue

            # Infer frequency and aggregate to monthly
            freq = self.freq_inferrer.infer_frequency(feat_df['value_date'])
            if freq != 'M':
                feat_df = self.freq_inferrer.aggregate_to_monthly(
                    feat_df, method=feat_spec.aggregation
                )
                # After aggregation, date is already the index
                feat_series = feat_df['value'].sort_index()
            else:
                feat_series = feat_df.set_index('value_date')['value'].sort_index()

            # Apply transformation
            feat_series = self.transformer.apply_transformation(
                feat_series, feat_spec.transformation,
                series_name=feat_spec.internal_series_name
            )

            # Handle missing values if specified
            if feat_spec.fill_method:
                if feat_spec.fill_method == 'ffill':
                    feat_series = feat_series.ffill()
                elif feat_spec.fill_method == 'bfill':
                    feat_series = feat_series.bfill()
                elif feat_spec.fill_method == 'interpolate':
                    feat_series = feat_series.interpolate()

            # Create lag features
            lag_features = LagFeatureBuilder.create_lag_features(
                series=feat_series,
                n_lags=feat_spec.n_lags,
                horizon=horizon,
                publication_lag=feat_spec.publication_lag,
                feature_name=feat_spec.internal_series_name
            )

            all_features.append(lag_features)

        # Combine all features
        if all_features:
            feature_df = pd.concat(all_features, axis=1)
            return feature_df
        else:
            return pd.DataFrame()
