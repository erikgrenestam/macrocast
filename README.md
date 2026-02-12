# Macrocast: BART-based Macroeconomic Forecasting

A production-ready system for forecasting macroeconomic indicators using Bayesian Additive Regression Trees (BART). Designed for direct multi-horizon forecasting with proper handling of data revisions, publication lags, and realistic backtesting.

## Features

- **BART Forecasting**: Bayesian Additive Regression Trees via `pymc-bart`
- **Multi-horizon**: Direct forecasting from 1 to 24 months ahead
- **Vintage Data**: Handles data revisions and creates pseudo-vintages for backtesting
- **Publication Lags**: Respects delayed data availability per variable
- **No Data Leakage**: Careful lag structure ensures realistic forecasts
- **Parallel MCMC**: Efficient sampling with multiple chains
- **Comprehensive Evaluation**: RMSE, MAE, MAPE metrics with visualizations

## Installation

```bash
# Clone and navigate to project
cd macrocast

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### 1. Configure Your Model

Edit `configs/spec.yaml` to specify:
- Target variable and transformation
- Feature variables with lags and publication delays
- Model hyperparameters (trees, chains, MCMC settings)
- Backtest configuration

Example:
```yaml
target:
  internal_series_name: "hicp_dk_dst"
  transformation: "12m_diff"  # Year-over-year inflation

features:
  - internal_series_name: "industrial_production_index_dk_dst"
    transformation: "12m_diff"
    n_lags: 6
    publication_lag: 2

model:
  n_trees: 50
  n_chains: 4
  horizons: [1, 3, 6, 12, 18, 24]
```

### 2. Run Backtest

```bash
python scripts/run_backtest.py \
    --config configs/spec.yaml \
    --data data/example_data.parquet \
    --output results/backtest_20260212
```

This will:
- Train BART models at each forecast date
- Generate forecasts for all horizons
- Compute evaluation metrics
- Create visualizations

### 3. Train Production Models

```bash
python scripts/train_model.py \
    --config configs/spec.yaml \
    --data data/example_data.parquet \
    --output models/production
```

### 4. Generate Forecasts

```bash
python scripts/generate_forecasts.py \
    --models models/production \
    --config configs/spec.yaml \
    --data data/example_data.parquet \
    --output results/latest_forecast.csv
```

## Project Structure

```
macrocast/
├── src/
│   ├── config.py           # Configuration management
│   ├── transform.py        # Data transformation pipeline
│   ├── model.py            # BART model training & inference
│   ├── eval.py             # Backtesting & evaluation
│   └── utils.py            # Utility functions
├── scripts/
│   ├── run_backtest.py     # Run backtest
│   ├── train_model.py      # Train models
│   └── generate_forecasts.py  # Generate forecasts
├── configs/
│   └── spec.yaml           # Model specification
├── data/
│   └── example_data.parquet  # Input data
├── models/                 # Saved models (created)
├── results/                # Output (created)
└── logs/                   # Logs (created)
```

## Data Format

Input data should be in parquet format with long structure:

| Column | Type | Description |
|--------|------|-------------|
| `value_date` | datetime | Date of observation |
| `value` | float | Numeric value |
| `internal_series_name` | string | Variable identifier |
| `valid_from` | datetime | Vintage start date (optional) |
| `valid_to` | datetime | Vintage end date (optional) |

## Key Concepts

### Direct Multi-Horizon Forecasting

For each horizon h, we train a separate model:
- h=1: Model predicts 1-month ahead
- h=6: Model predicts 6-months ahead
- h=12: Model predicts 12-months ahead

This avoids error accumulation from iterative forecasting.

### Lag Structure & Publication Lags

To prevent data leakage, the first available lag is:
```
first_lag = horizon + publication_lag
```

Example: For h=6 with publication_lag=2:
- First lag is L8 (using data from t-8 to forecast t+6)
- Ensures data at t-8 was published by time t-2

### Data Transformations

Supports:
- `none`: No transformation
- `log`: Natural logarithm
- `diff`: First difference
- `12m_diff`: 12-month difference (for YoY rates)
- `log_diff`: Log then difference

Use stationary transformations (like `12m_diff` for inflation) to avoid BART's extrapolation limitations.

### Pseudo-Vintages

For realistic backtesting, the system creates pseudo-vintages that reconstruct what data would have been available at each historical forecast date, respecting:
- Data revisions (valid_from/valid_to timestamps)
- Publication lags per variable

## Configuration Reference

### Target Specification

```yaml
target:
  internal_series_name: string   # Variable identifier
  transformation: string          # none|log|diff|12m_diff|log_diff
  detrend: bool                   # Apply detrending (default: false)
  detrend_method: string          # linear|hp_filter|moving_average
```

### Feature Specification

```yaml
features:
  - internal_series_name: string  # Variable identifier
    aggregation: string           # mean|sd|first|last (for high-freq data)
    transformation: string        # none|log|diff|12m_diff|log_diff
    n_lags: int                   # Number of lags to include
    publication_lag: int          # Publication delay in months
    fill_method: string           # ffill|bfill|interpolate|null
```

### Model Specification

```yaml
model:
  n_trees: int                    # Number of trees in BART ensemble
  n_chains: int                   # Number of MCMC chains
  n_cores: int                    # CPU cores for parallel sampling
  n_tune: int                     # Warmup iterations
  n_draws: int                    # Posterior samples
  horizons: list[int]             # Forecast horizons in months
```

### Backtest Specification

```yaml
backtest:
  method: string                  # expanding|rolling
  initial_window: int             # Minimum training months
  rolling_window: int             # Window size for rolling (if applicable)
  step_size: int                  # Months between forecasts
  start_date: string              # Start date (YYYY-MM-DD)
  end_date: string                # End date (YYYY-MM-DD)
```

## Output Files

### Backtest Results

- `forecasts.csv`: All forecasts with actuals and errors
- `metrics.csv`: RMSE, MAE, MAPE by horizon
- `error_plot.png`: Time series of errors
- `metrics_plot.png`: Bar charts of metrics
- `actual_vs_forecast.png`: Scatter plots

### Trained Models

- `model_h{horizon}_idata.nc`: Inference data per horizon
- `variable_importance_h{horizon}.csv`: Feature importance
- `config.pkl`: Configuration used for training

## Performance Tips

1. **Reduce MCMC iterations**: For quick tests, use smaller `n_tune` and `n_draws`
2. **Fewer horizons**: Start with 1-2 horizons before full set
3. **Limit features**: Begin with key predictors
4. **Use fewer chains**: 2 chains for testing, 4+ for production
5. **Smaller trees**: Reduce `n_trees` for faster training

## Troubleshooting

### Insufficient Data

If you see "insufficient data" warnings:
- Reduce `initial_window` in backtest config
- Check if features exist in your data
- Verify date ranges align

### Convergence Issues

If R-hat > 1.01 or low ESS:
- Increase `n_tune` and `n_draws`
- Use more chains (4+)
- Check for data issues (NaNs, extreme values)

### Memory Issues

For large datasets:
- Reduce `n_lags` per feature
- Use fewer posterior samples
- Process horizons sequentially

## Dependencies

- Python ≥3.13
- pandas ≥3.0.0
- pymc ≥5.27.1
- pymc-bart ≥0.11.0
- matplotlib ≥3.10.8
- pyarrow ≥19.0.0
- pyyaml ≥6.0.0
- arviz ≥0.21.0
- scikit-learn ≥1.6.0

## Citation

If you use this system in research, please cite:

```
Macrocast: BART-based Macroeconomic Forecasting
https://github.com/yourusername/macrocast
```

## License

MIT License

## Support

For issues and questions:
- GitHub Issues: [Report an issue](https://github.com/yourusername/macrocast/issues)
- Documentation: See CLAUDE.md for development notes
