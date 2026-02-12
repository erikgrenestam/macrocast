"""Backtesting and evaluation for BART forecasting models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from config import ModelConfig
from model import BARTForecaster, ForecastResult
from transform import DataTransformer


@dataclass
class BacktestResult:
    """Container for backtest results."""

    forecasts: pd.DataFrame  # Columns: forecast_date, horizon, forecast, actual, error
    metrics: pd.DataFrame  # Columns: horizon, rmse, mae, mape
    forecast_dates: List[pd.Timestamp]
    config: ModelConfig


class BacktestEngine:
    """Perform expanding or rolling window backtests."""

    def __init__(
        self,
        config: ModelConfig,
        data_path: Path
    ):
        self.config = config
        self.data_transformer = DataTransformer(config, data_path)
        self.results: Optional[BacktestResult] = None

    def run_backtest(self) -> BacktestResult:
        """
        Execute full backtest across all pseudo-vintages.

        Process:
        1. Create list of forecast dates based on backtest config
        2. For each forecast date:
           a. Create pseudo-vintage (data as of that date)
           b. Determine training window
           c. Train models for all horizons
           d. Generate forecasts
           e. Store forecasts
        3. Evaluate against realized values
        4. Compute metrics
        """
        print("Starting backtest...")
        print(f"  Method: {self.config.backtest.method}")
        print(f"  Period: {self.config.backtest.start_date} to {self.config.backtest.end_date}")

        # Generate forecast dates
        forecast_dates = self._generate_forecast_dates()
        print(f"  Number of forecast origins: {len(forecast_dates)}")

        # Storage for results
        all_forecasts = []

        # Iterate through forecast dates
        for i, forecast_date in enumerate(forecast_dates):
            print(f"\n[{i+1}/{len(forecast_dates)}] Forecasting from {forecast_date.date()}")

            try:
                # Train models and generate forecasts
                horizon_forecasts = self._forecast_at_date(forecast_date, i)
                all_forecasts.extend(horizon_forecasts)

            except Exception as e:
                print(f"  Error: {e}")
                continue

        # Convert to DataFrame
        forecasts_df = pd.DataFrame(all_forecasts)

        if len(forecasts_df) == 0:
            raise ValueError("No forecasts generated. Check data availability and configuration.")

        # Merge with actuals
        forecasts_df = self._merge_actuals(forecasts_df)

        # Compute metrics
        metrics_df = self._compute_metrics(forecasts_df)

        # Create result object
        self.results = BacktestResult(
            forecasts=forecasts_df,
            metrics=metrics_df,
            forecast_dates=forecast_dates,
            config=self.config
        )

        print("\n" + "="*60)
        print("Backtest complete!")
        print("="*60)
        print("\nMetrics by horizon:")
        print(metrics_df.to_string(index=False))

        return self.results

    def _forecast_at_date(
        self,
        forecast_date: pd.Timestamp,
        iteration: int
    ) -> List[Dict]:
        """
        Generate forecasts for all horizons at a specific date.

        Args:
            forecast_date: Date to generate forecast from
            iteration: Which iteration of backtest (for initial window check)

        Returns: List of forecast dictionaries
        """
        forecasts = []

        # Create forecaster for this date
        forecaster = BARTForecaster(self.config)

        for horizon in self.config.model.horizons:
            print(f"  Horizon {horizon}...", end=" ", flush=True)

            try:
                # Prepare data as of this date
                data = self.data_transformer.prepare_data(
                    horizon=horizon,
                    as_of_date=forecast_date
                )

                if len(data.X) < self.config.backtest.initial_window:
                    print(f"insufficient data (need {self.config.backtest.initial_window}, have {len(data.X)})")
                    continue

                # Determine training window
                train_X, train_y = self._get_training_window(data, iteration)

                # Train BART model
                forecaster.train(
                    X=train_X,
                    y=train_y,
                    horizon=horizon,
                    feature_names=None  # Skip variable importance for speed
                )

                # Generate forecast for latest observation
                X_latest = data.X[-1:, :]
                point_fc, lower, upper = forecaster.predict(X_latest, horizon=horizon)

                # Store forecast
                forecasts.append({
                    'forecast_date': forecast_date,
                    'horizon': horizon,
                    'forecast': point_fc[0],
                    'lower_bound': lower[0],
                    'upper_bound': upper[0],
                    'target_date': forecast_date + pd.DateOffset(months=horizon)
                })

                print("done")

            except Exception as e:
                print(f"error: {e}")
                continue

        return forecasts

    def _generate_forecast_dates(self) -> List[pd.Timestamp]:
        """Generate list of forecast origin dates."""
        start = pd.Timestamp(self.config.backtest.start_date)
        end = pd.Timestamp(self.config.backtest.end_date)
        step = self.config.backtest.step_size

        dates = pd.date_range(
            start=start,
            end=end,
            freq=f'{step}MS'  # Month start
        )

        return dates.tolist()

    def _get_training_window(
        self,
        data,
        iteration: int
    ):
        """
        Determine training window based on backtest method.

        Args:
            data: TransformedData object
            iteration: Which backtest iteration

        Returns: (train_X, train_y) arrays
        """
        if self.config.backtest.method == 'expanding':
            # Expanding window: use all available data
            return data.X, data.y

        elif self.config.backtest.method == 'rolling':
            # Rolling window: fixed window size
            if self.config.backtest.rolling_window is None:
                raise ValueError("rolling_window must be specified for rolling backtest")

            window_size = self.config.backtest.rolling_window

            if len(data.X) > window_size:
                return data.X[-window_size:], data.y[-window_size:]
            else:
                return data.X, data.y

        else:
            raise ValueError(f"Unknown backtest method: {self.config.backtest.method}")

    def _merge_actuals(self, forecasts_df: pd.DataFrame) -> pd.DataFrame:
        """Merge forecast DataFrame with actual realized values."""
        # Load full deduplicated data
        raw_df = self.data_transformer.loader.load()
        dedup_df = self.data_transformer.loader.deduplicate(raw_df)

        # Prepare target series
        target_series = self.data_transformer._prepare_target(dedup_df)

        # Merge actuals by target_date
        actuals = target_series.reset_index()
        actuals.columns = ['target_date', 'actual']

        # Convert to datetime if needed
        forecasts_df['target_date'] = pd.to_datetime(forecasts_df['target_date'])
        actuals['target_date'] = pd.to_datetime(actuals['target_date'])

        forecasts_merged = forecasts_df.merge(
            actuals,
            on='target_date',
            how='left'
        )

        # Compute errors
        forecasts_merged['error'] = (
            forecasts_merged['forecast'] - forecasts_merged['actual']
        )
        forecasts_merged['abs_error'] = forecasts_merged['error'].abs()
        forecasts_merged['squared_error'] = forecasts_merged['error'] ** 2

        return forecasts_merged

    def _compute_metrics(self, forecasts_df: pd.DataFrame) -> pd.DataFrame:
        """Compute evaluation metrics by horizon."""
        metrics = []

        for horizon in self.config.model.horizons:
            horizon_data = forecasts_df[
                forecasts_df['horizon'] == horizon
            ].dropna(subset=['actual'])

            if len(horizon_data) == 0:
                continue

            rmse = np.sqrt(horizon_data['squared_error'].mean())
            mae = horizon_data['abs_error'].mean()

            # MAPE (handle division by zero)
            mape_values = (
                horizon_data['abs_error'] / horizon_data['actual'].abs()
            )
            mape = mape_values[np.isfinite(mape_values)].mean() * 100

            metrics.append({
                'horizon': horizon,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'n_forecasts': len(horizon_data)
            })

        return pd.DataFrame(metrics)


class BacktestVisualizer:
    """Create visualizations of backtest results."""

    def __init__(self, results: BacktestResult):
        self.results = results

    def plot_forecast_errors(self, save_path: Optional[Path] = None):
        """Plot forecast errors over time for each horizon."""
        horizons = sorted(self.results.forecasts['horizon'].unique())
        n_horizons = len(horizons)

        fig, axes = plt.subplots(
            n_horizons,
            1,
            figsize=(12, 3 * n_horizons),
            sharex=True
        )

        if n_horizons == 1:
            axes = [axes]

        for ax, horizon in zip(axes, horizons):
            data = self.results.forecasts[
                (self.results.forecasts['horizon'] == horizon) &
                (self.results.forecasts['actual'].notna())
            ]

            ax.plot(data['forecast_date'], data['error'], 'o-', alpha=0.7, markersize=4)
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel(f'Error (h={horizon})')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Forecast Date')
        fig.suptitle('Forecast Errors by Horizon', fontsize=14, y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_metrics_by_horizon(self, save_path: Optional[Path] = None):
        """Plot RMSE and MAE by horizon."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        metrics = self.results.metrics

        ax1.bar(metrics['horizon'], metrics['rmse'], color='steelblue')
        ax1.set_xlabel('Horizon (months)')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Root Mean Squared Error')
        ax1.grid(True, alpha=0.3, axis='y')

        ax2.bar(metrics['horizon'], metrics['mae'], color='coral')
        ax2.set_xlabel('Horizon (months)')
        ax2.set_ylabel('MAE')
        ax2.set_title('Mean Absolute Error')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_forecast_fan_chart(
        self,
        forecast_date: pd.Timestamp,
        save_path: Optional[Path] = None
    ):
        """
        Create fan chart showing forecast with uncertainty bands.

        Args:
            forecast_date: Which forecast to visualize
        """
        # Filter to selected forecast date
        fc_data = self.results.forecasts[
            self.results.forecasts['forecast_date'] == forecast_date
        ].sort_values('horizon')

        if len(fc_data) == 0:
            print(f"No forecasts found for {forecast_date}")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create dates for forecast
        target_dates = pd.to_datetime(fc_data['target_date'])

        # Plot forecast
        ax.plot(target_dates, fc_data['forecast'], 'o-', label='Forecast', color='blue', linewidth=2)

        # Plot uncertainty bands
        ax.fill_between(
            target_dates,
            fc_data['lower_bound'],
            fc_data['upper_bound'],
            alpha=0.3,
            color='blue',
            label='90% Interval'
        )

        # Plot actuals (if available)
        actual_data = fc_data.dropna(subset=['actual'])
        if len(actual_data) > 0:
            ax.plot(
                pd.to_datetime(actual_data['target_date']),
                actual_data['actual'],
                's-',
                label='Actual',
                color='red',
                linewidth=2,
                markersize=6
            )

        ax.set_xlabel('Date')
        ax.set_ylabel(self.results.config.target.internal_series_name)
        ax.set_title(f'Forecast from {forecast_date.date()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_actual_vs_forecast(self, save_path: Optional[Path] = None):
        """Plot actual vs forecast scatter plot for each horizon."""
        horizons = sorted(self.results.forecasts['horizon'].unique())
        n_horizons = len(horizons)

        n_cols = min(3, n_horizons)
        n_rows = (n_horizons + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, horizon in enumerate(horizons):
            ax = axes[idx]
            data = self.results.forecasts[
                (self.results.forecasts['horizon'] == horizon) &
                (self.results.forecasts['actual'].notna())
            ]

            ax.scatter(data['actual'], data['forecast'], alpha=0.5, s=30)

            # Add 45-degree line
            min_val = min(data['actual'].min(), data['forecast'].min())
            max_val = max(data['actual'].max(), data['forecast'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

            ax.set_xlabel('Actual')
            ax.set_ylabel('Forecast')
            ax.set_title(f'Horizon {horizon} months')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_horizons, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
