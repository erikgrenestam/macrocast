"""BART model training and forecasting."""

import pymc as pm
import pymc_bart as pmb
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from config import ModelConfig
from transform import DataTransformer, TransformedData


@dataclass
class ForecastResult:
    """Container for forecast results."""

    point_forecast: np.ndarray  # Shape: (n_horizons,)
    lower_bound: np.ndarray  # 5th percentile
    upper_bound: np.ndarray  # 95th percentile
    samples: np.ndarray  # Full posterior samples
    horizons: list[int]
    forecast_date: pd.Timestamp


class BARTForecaster:
    """BART-based forecaster using direct multi-step strategy."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.models: Dict[int, Tuple] = {}  # horizon -> (model, idata)
        self.variable_importance: Dict[int, pd.DataFrame] = {}

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        horizon: int,
        feature_names: Optional[list[str]] = None
    ) -> None:
        """
        Train BART model for a specific forecast horizon.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            horizon: Forecast horizon this model is for
            feature_names: Names of features (for variable importance)
        """
        print(f"\nTraining BART model for h={horizon}...")
        print(f"  Data shape: X={X.shape}, y={y.shape}")

        # Build PyMC model with pm.Data for out-of-sample predictions
        with pm.Model() as model:
            # Wrap X in pm.Data for later prediction
            X_data = pm.Data('X', X)

            # Create BART model
            mu = pmb.BART(
                'mu',
                X=X_data,
                Y=y,
                m=self.config.model.n_trees
            )

            # Likelihood (Gaussian errors for regression)
            sigma = pm.HalfNormal('sigma', sigma=1)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

            # Sample posterior with parallel chains
            print(f"  Sampling {self.config.model.n_chains} chains in parallel...")
            idata = pm.sample(
                draws=self.config.model.n_draws,
                tune=self.config.model.n_tune,
                chains=self.config.model.n_chains,
                cores=self.config.model.n_cores,
                return_inferencedata=True,
                random_seed=42
            )

        # Store model artifacts
        self.models[horizon] = (model, idata)

        # Print convergence summary
        print(f"  Training complete.")
        summary = pm.summary(idata, var_names=['sigma'])
        print(f"  Sigma: mean={summary['mean'].values[0]:.4f}, "
              f"r_hat={summary['r_hat'].values[0]:.4f}")

        # Compute variable importance (simplified)
        if feature_names:
            var_imp = self._compute_variable_importance(idata, feature_names)
            self.variable_importance[horizon] = var_imp

    def predict(
        self,
        X_new: np.ndarray,
        horizon: int,
        return_samples: bool = False
    ) -> Tuple:
        """
        Generate predictions for new data.

        Args:
            X_new: Feature matrix for prediction (n_samples, n_features)
            horizon: Which horizon model to use
            return_samples: If True, return full posterior samples

        Returns:
            (point_forecast, lower_bound, upper_bound)
            If return_samples=True, also returns samples array

        Note: This is a simplified MVP implementation. For production, proper BART
        out-of-sample prediction should use the fitted tree structure with new X.
        """
        if horizon not in self.models:
            raise ValueError(f"No trained model for horizon {horizon}")

        model, idata = self.models[horizon]

        # Extract posterior predictions from training
        # For MVP: use posterior statistics from the model
        mu_posterior = idata.posterior['mu'].values  # (chains, draws, n_train)
        sigma_posterior = idata.posterior['sigma'].values  # (chains, draws)

        # Flatten chains and draws
        mu_flat = mu_posterior.reshape(-1, mu_posterior.shape[-1])  # (total_samples, n_train)
        sigma_flat = sigma_posterior.flatten()  # (total_samples,)

        # Use the mean prediction from training as baseline
        # This is a simplified approach for MVP
        baseline_prediction = mu_flat.mean()

        # Generate prediction samples by sampling from Normal(baseline, sigma)
        n_posterior_samples = len(sigma_flat)
        y_pred_samples = np.random.normal(
            baseline_prediction,
            sigma_flat[:, np.newaxis],
            size=(n_posterior_samples, X_new.shape[0])
        )

        # Compute statistics
        point_forecast = y_pred_samples.mean(axis=0)
        lower_bound = np.percentile(y_pred_samples, 5, axis=0)
        upper_bound = np.percentile(y_pred_samples, 95, axis=0)

        if return_samples:
            return point_forecast, lower_bound, upper_bound, y_pred_samples
        else:
            return point_forecast, lower_bound, upper_bound

    def forecast_all_horizons(
        self,
        X_latest: np.ndarray,
        forecast_date: pd.Timestamp
    ) -> ForecastResult:
        """
        Generate forecasts for all trained horizons.

        Args:
            X_latest: Latest feature data (single observation)
            forecast_date: Date of the forecast

        Returns: ForecastResult with forecasts for all horizons
        """
        horizons = sorted(self.models.keys())

        point_forecasts = []
        lower_bounds = []
        upper_bounds = []
        all_samples = []

        for h in horizons:
            pt, lb, ub, samples = self.predict(
                X_latest.reshape(1, -1),
                horizon=h,
                return_samples=True
            )
            point_forecasts.append(pt[0])
            lower_bounds.append(lb[0])
            upper_bounds.append(ub[0])
            all_samples.append(samples[:, 0])

        return ForecastResult(
            point_forecast=np.array(point_forecasts),
            lower_bound=np.array(lower_bounds),
            upper_bound=np.array(upper_bounds),
            samples=np.array(all_samples),
            horizons=horizons,
            forecast_date=forecast_date
        )

    def _compute_variable_importance(
        self,
        idata,
        feature_names: list[str]
    ) -> pd.DataFrame:
        """
        Compute variable importance from BART split counts.

        BART variable importance is based on how often each variable
        is used in splitting decisions across all trees.

        Simplified implementation for MVP.
        """
        # Placeholder: uniform importance
        # Full implementation would extract split counts from BART posterior
        n_features = len(feature_names)
        importance = np.ones(n_features) / n_features

        var_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return var_imp_df

    def save(self, path: Path, horizon: Optional[int] = None) -> None:
        """
        Save trained models to disk.

        Args:
            path: Directory to save models
            horizon: If specified, save only this horizon. Else save all.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        horizons_to_save = [horizon] if horizon else self.models.keys()

        for h in horizons_to_save:
            if h not in self.models:
                continue

            model, idata = self.models[h]

            # Save inference data
            idata.to_netcdf(path / f"model_h{h}_idata.nc")

            # Save variable importance
            if h in self.variable_importance:
                self.variable_importance[h].to_csv(
                    path / f"variable_importance_h{h}.csv",
                    index=False
                )

        # Save config
        with open(path / "config.pkl", "wb") as f:
            pickle.dump(self.config, f)

        print(f"Models saved to {path}")

    def load(self, path: Path, horizon: Optional[int] = None) -> None:
        """
        Load trained models from disk.

        Args:
            path: Directory containing saved models
            horizon: If specified, load only this horizon. Else load all.
        """
        import arviz as az

        path = Path(path)

        if horizon:
            horizons_to_load = [horizon]
        else:
            # Find all saved models
            model_files = list(path.glob("model_h*_idata.nc"))
            horizons_to_load = [
                int(f.stem.split('_')[1][1:]) for f in model_files
            ]

        for h in horizons_to_load:
            idata_path = path / f"model_h{h}_idata.nc"
            if not idata_path.exists():
                continue

            # Load inference data
            idata = az.from_netcdf(idata_path)

            # Note: Full model reconstruction for predictions requires
            # recreating the PyMC model structure, which is complex with BART
            # For now, we load idata only for diagnostics
            self.models[h] = (None, idata)

            # Load variable importance
            var_imp_path = path / f"variable_importance_h{h}.csv"
            if var_imp_path.exists():
                self.variable_importance[h] = pd.read_csv(var_imp_path)

        print(f"Loaded {len(horizons_to_load)} models from {path}")


class ProductionForecaster:
    """High-level interface for production forecasting."""

    def __init__(
        self,
        config: ModelConfig,
        data_path: Path,
        model_dir: Optional[Path] = None
    ):
        self.config = config
        self.data_transformer = DataTransformer(config, data_path)
        self.forecaster = BARTForecaster(config)

        if model_dir and Path(model_dir).exists():
            self.forecaster.load(model_dir)

    def train_all_horizons(self, save_path: Optional[Path] = None) -> None:
        """Train models for all specified forecast horizons."""
        for horizon in self.config.model.horizons:
            print(f"\n{'='*60}")
            print(f"Training horizon: {horizon} months")
            print('='*60)

            # Prepare data for this horizon
            data = self.data_transformer.prepare_data(horizon=horizon)

            # Train BART model
            self.forecaster.train(
                X=data.X,
                y=data.y,
                horizon=horizon,
                feature_names=data.feature_names
            )

        # Save models
        if save_path:
            self.forecaster.save(save_path)

    def generate_latest_forecast(self) -> ForecastResult:
        """Generate forecast using most recent available data."""
        # Use first horizon to get latest data
        # (all horizons use same X, just different lags)
        first_horizon = min(self.config.model.horizons)
        data = self.data_transformer.prepare_data(horizon=first_horizon)

        # Get latest observation
        X_latest = data.X[-1:, :]
        forecast_date = data.dates[-1]

        # Generate forecasts for all horizons
        return self.forecaster.forecast_all_horizons(X_latest, forecast_date)
