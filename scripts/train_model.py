"""Train production BART forecasting models."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ModelConfig
from model import ProductionForecaster
from utils import setup_logging, ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description='Train BART forecasting models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        help='Output directory for trained models'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('logs'),
        help='Directory for log files'
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.log_dir, name='training')
    output_dir = ensure_dir(args.output)

    logger.info(f"Loading configuration from {args.config}")
    config = ModelConfig.from_yaml(args.config)
    config.validate()

    logger.info(f"Configuration loaded:")
    logger.info(f"  Target: {config.target.internal_series_name}")
    logger.info(f"  Features: {len(config.features)}")
    logger.info(f"  Horizons: {config.model.horizons}")
    logger.info(f"  Model: {config.model.n_trees} trees, {config.model.n_chains} chains")

    # Create forecaster
    logger.info("Initializing production forecaster...")
    forecaster = ProductionForecaster(config, args.data)

    # Train models for all horizons
    logger.info("Training models for all horizons...")
    forecaster.train_all_horizons(save_path=output_dir)

    logger.info(f"\nModels saved to: {output_dir}")
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"\nModels saved to: {output_dir}")
    print(f"Horizons trained: {config.model.horizons}")


if __name__ == '__main__':
    main()
