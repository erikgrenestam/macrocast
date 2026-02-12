"""Data transformation pipeline for BART forecasting."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
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


class DataLoader:
    """Load and deduplicate raw parquet data."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self._raw_data: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load raw data from parquet."""
        df = pd.read_parquet(self.data_path)
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

    @staticmethod
    def apply_transformation(
        series: pd.Series,
        transformation: str
    ) -> pd.Series:
        """
        Apply specified transformation.

        Returns: Transformed series (may be shorter due to differencing)
        """
        if transformation == 'none':
            return series
        elif transformation == 'log':
            return np.log(series)
        elif transformation == 'diff':
            return series.diff()
        elif transformation == '12m_diff':
            return series.diff(12)
        elif transformation == 'log_diff':
            return np.log(series).diff()
        else:
            raise ValueError(f"Unknown transformation: {transformation}")

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
        else:
            raise ValueError(f"Unknown transformation: {transformation}")


class TrendRemover:
    """Handle trend extraction for BART (which cannot extrapolate trends)."""

    def __init__(self, method: str = 'linear'):
        self.method = method
        self.trend_params = {}

    def fit_trend(self, y: pd.Series) -> pd.Series:
        """
        Fit trend to series and return trend component.
        """
        if self.method == 'linear':
            # Fit linear trend
            t = np.arange(len(y))
            coeffs = np.polyfit(t, y.values, deg=1)
            self.trend_params = {'slope': coeffs[0], 'intercept': coeffs[1], 'length': len(y)}
            trend = coeffs[0] * t + coeffs[1]
            return pd.Series(trend, index=y.index)

        elif self.method == 'hp_filter':
            # Hodrick-Prescott filter
            try:
                from statsmodels.tsa.filters.hp_filter import hpfilter
                cycle, trend = hpfilter(y, lamb=129600)  # Monthly lambda
                self.trend_params = {'trend_series': trend}
                return trend
            except ImportError:
                raise ImportError("statsmodels required for hp_filter method")

        elif self.method == 'moving_average':
            # Centered moving average
            window = 24  # 2-year window for monthly data
            trend = y.rolling(window=window, center=True).mean()
            # Fill edges with simple trend
            trend = trend.bfill().ffill()
            self.trend_params = {'trend_series': trend}
            return trend

        else:
            raise ValueError(f"Unknown detrend method: {self.method}")

    def remove_trend(self, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Remove trend from series.

        Returns: (detrended_series, trend_component)
        """
        trend = self.fit_trend(y)
        detrended = y - trend
        return detrended, trend

    def extrapolate_trend(self, n_periods: int) -> np.ndarray:
        """
        Extrapolate trend for forecast horizons.

        Args:
            n_periods: Number of periods to extrapolate

        Returns: Array of length n_periods with trend values
        """
        if self.method == 'linear':
            # Linear extrapolation is straightforward
            last_t = self.trend_params.get('length', 0)
            future_t = np.arange(last_t, last_t + n_periods)
            trend = (
                self.trend_params['slope'] * future_t +
                self.trend_params['intercept']
            )
            return trend

        elif self.method in ['hp_filter', 'moving_average']:
            # Extrapolate last trend value (assumes trend persists)
            last_trend = self.trend_params['trend_series'].iloc[-1]
            return np.full(n_periods, last_trend)

        else:
            raise ValueError(f"Unknown detrend method: {self.method}")


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
        self.trend_remover = None

        if config.target.detrend:
            self.trend_remover = TrendRemover(config.target.detrend_method)

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
        # 1. Load and deduplicate
        raw_df = self.loader.load()

        if as_of_date:
            df = self.loader.create_vintage(raw_df, as_of_date)
        else:
            df = self.loader.deduplicate(raw_df)

        # 2. Prepare target variable
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
            }
        )

    def _prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Prepare target variable with transformations and optional detrending."""
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
            target_series, target_spec.transformation
        )

        # Detrend if specified
        if target_spec.detrend and self.trend_remover:
            target_series, _ = self.trend_remover.remove_trend(target_series)

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
                feat_series, feat_spec.transformation
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
