"""Utility functions for the macrocast project."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        path: Path to directory

    Returns:
        Path object for the directory
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(log_dir: Path, name: str = 'macrocast', level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    log_dir = ensure_dir(log_dir)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'{name}.log'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(name)


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    prefix: str = 'results'
) -> None:
    """
    Save results dictionary to files.

    Args:
        results: Dictionary of results to save
        output_dir: Directory to save results
        prefix: Filename prefix
    """
    output_dir = ensure_dir(output_dir)

    for name, data in results.items():
        filepath = output_dir / f"{prefix}_{name}"

        if isinstance(data, pd.DataFrame):
            data.to_csv(f"{filepath}.csv", index=False)
        elif isinstance(data, dict):
            pd.DataFrame([data]).to_csv(f"{filepath}.csv", index=False)
        elif isinstance(data, np.ndarray):
            np.save(f"{filepath}.npy", data)


def format_date(dt: pd.Timestamp) -> str:
    """
    Format timestamp for filenames.

    Args:
        dt: Timestamp to format

    Returns:
        Date string in YYYYMMDD format
    """
    return dt.strftime('%Y%m%d')
