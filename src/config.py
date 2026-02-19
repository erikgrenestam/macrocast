"""Configuration management for BART forecasting models."""

from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path
import yaml


@dataclass
class VariableSpec:
    """Specification for a single input variable."""

    internal_series_name: str
    aggregation: Literal['mean', 'sd', 'first', 'last'] = 'last'
    transformation: Literal['none', 'log', 'diff', '12m_diff', 'log_diff', 'log_12m_diff'] = 'none'
    n_lags: int = 12
    publication_lag: int = 1
    fill_method: Optional[Literal['ffill', 'bfill', 'interpolate']] = None


@dataclass
class TargetSpec:
    """Specification for target variable."""

    internal_series_name: str
    transformation: Literal['none', 'log', 'diff', '12m_diff', 'log_diff', 'log_12m_diff'] = 'none'


@dataclass
class ModelSpec:
    """BART model hyperparameters."""

    n_trees: int = 50
    n_chains: int = 4
    n_cores: int = 4
    n_tune: int = 1000
    n_draws: int = 1000
    horizons: list[int] = field(default_factory=lambda: [1, 3, 6, 12])


@dataclass
class BacktestSpec:
    """Backtesting configuration."""

    method: Literal['expanding', 'rolling'] = 'expanding'
    initial_window: int = 120
    rolling_window: Optional[int] = None
    step_size: int = 1
    start_date: str = '2010-01-01'
    end_date: str = '2024-12-31'


@dataclass
class ModelConfig:
    """Complete model configuration."""

    target: TargetSpec
    features: list[VariableSpec] = field(default_factory=list)
    model: ModelSpec = field(default_factory=ModelSpec)
    backtest: BacktestSpec = field(default_factory=BacktestSpec)

    @classmethod
    def from_yaml(cls, path: Path) -> 'ModelConfig':
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        target = TargetSpec(**data['target'])
        features = [VariableSpec(**v) for v in data.get('features', [])]
        model = ModelSpec(**data.get('model', {}))
        backtest = BacktestSpec(**data.get('backtest', {}))

        return cls(target=target, features=features, model=model, backtest=backtest)

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.features:
            # Ensure target is not in features
            feature_names = {f.internal_series_name for f in self.features}
            if self.target.internal_series_name in feature_names:
                raise ValueError("Target cannot be in feature list")

            # Validate that first lag + n_lags is reasonable
            # first_lag = horizon + publication_lag
            # We need at least first_lag + n_lags historical periods
            if len(self.model.horizons) > 0:
                max_horizon = max(self.model.horizons)
                for feature in self.features:
                    max_lag_needed = max_horizon + feature.publication_lag + feature.n_lags - 1
                    if max_lag_needed > 240:  # Warn if need more than 20 years of data
                        import warnings
                        warnings.warn(
                            f"Feature {feature.internal_series_name} requires lag {max_lag_needed} "
                            f"for horizon {max_horizon}, which may exceed available data."
                        )

            # Validate n_lags are positive for all features
            if any(f.n_lags <= 0 for f in self.features):
                raise ValueError("All n_lags must be positive integers")

        # Validate backtest method
        if self.backtest.method == 'rolling' and self.backtest.rolling_window is None:
            raise ValueError("rolling_window must be specified when method='rolling'")

        # Validate horizons are positive
        if any(h <= 0 for h in self.model.horizons):
            raise ValueError("All horizons must be positive integers")
