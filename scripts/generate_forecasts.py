"""Generate latest forecasts using trained BART models."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ModelConfig
from model import ProductionForecaster
from utils import setup_logging, ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description='Generate forecasts using trained BART models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--models',
        type=Path,
        required=True,
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to spec.yaml configuration file'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Path to input parquet data file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output CSV file for forecasts'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('logs'),
        help='Directory for log files'
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.log_dir, name='forecast')
    ensure_dir(args.output.parent)

    logger.info(f"Loading configuration from {args.config}")
    config = ModelConfig.from_yaml(args.config)
    config.validate()

    # Load trained models
    logger.info(f"Loading trained models from {args.models}")
    forecaster = ProductionForecaster(config, args.data, model_dir=args.models)

    # Generate latest forecasts
    logger.info("Generating forecasts using latest available data...")
    forecast_result = forecaster.generate_latest_forecast()

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'forecast_date': forecast_result.forecast_date,
        'horizon': forecast_result.horizons,
        'target_date': [
            forecast_result.forecast_date + pd.DateOffset(months=h)
            for h in forecast_result.horizons
        ],
        'forecast': forecast_result.point_forecast,
        'lower_bound': forecast_result.lower_bound,
        'upper_bound': forecast_result.upper_bound
    })

    # Save forecasts
    logger.info(f"Saving forecasts to {args.output}")
    forecast_df.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"Forecast generated!")
    print(f"{'='*60}")
    print(f"\nForecast date: {forecast_result.forecast_date.date()}")
    print(f"\nForecasts:")
    print(forecast_df.to_string(index=False))
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
