"""Run BART backtest with specified configuration."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from config import ModelConfig
from evaluate import BacktestEngine, BacktestVisualizer
from model import BARTForecaster


def main():
    parser = argparse.ArgumentParser(
        description='Run BART forecasting backtest',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to spec.yaml configuration file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('logs'),
        help='Directory for log files'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualizations'
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.log_dir, name='backtest')
    output_dir = ensure_dir(args.output)

    logger.info(f"Loading configuration from {args.config}")
    config = ModelConfig.from_yaml(args.config)
    config.validate()

    logger.info(f"Configuration loaded:")
    logger.info(f"  Target: {config.target.internal_series_name}")
    logger.info(f"  Features: {len(config.features)}")
    logger.info(f"  Horizons: {config.model.horizons}")
    logger.info(f"  Backtest period: {config.backtest.start_date} to {config.backtest.end_date}")

    # Run backtest
    logger.info("Starting backtest...")
    engine = BacktestEngine(
        config=config,
        forecaster_cls=BARTForecaster,
        forecaster_kwargs={'config': config},
    )
    results = engine.run_backtest()

    # Save results
    logger.info(f"Saving results to {output_dir}")
    results.forecasts.to_csv(output_dir / 'forecasts.csv', index=False)
    results.metrics.to_csv(output_dir / 'metrics.csv', index=False)

    # Generate plots
    if not args.no_plots:
        logger.info("Generating visualizations...")
        viz = BacktestVisualizer(results)

        try:
            viz.plot_forecast_errors(output_dir / 'error_plot.png')
            logger.info("  - error_plot.png")
        except Exception as e:
            logger.error(f"  Error plotting forecast errors: {e}")

        try:
            viz.plot_metrics_by_horizon(output_dir / 'metrics_plot.png')
            logger.info("  - metrics_plot.png")
        except Exception as e:
            logger.error(f"  Error plotting metrics: {e}")

        try:
            viz.plot_actual_vs_forecast(output_dir / 'actual_vs_forecast.png')
            logger.info("  - actual_vs_forecast.png")
        except Exception as e:
            logger.error(f"  Error plotting actual vs forecast: {e}")

    logger.info("Backtest complete!")
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\nMetrics summary:")
    print(results.metrics.to_string(index=False))


if __name__ == '__main__':
    main()
