"""
services/analytics.py - Analysis and metrics utilities for FENRIR.

This module provides numerical tools for post-processing simulation
data such as missile–target encounters, trajectory logs, and
telemetry recorded from the Streamlit dashboard.

Future functionality may include:
- Monte Carlo campaign automation
- Hit probability estimation
- Miss distance statistics
- Performance envelopes (N vs. amax trade-offs)
- Data smoothing and filtering utilities

Author: Matteo Da Venezia
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_miss_distance(missile_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """
    Compute the Euclidean miss distance between missile and target positions.

    Parameters
    ----------
    missile_pos : np.ndarray
        2D position vector of the missile [x, y].
    target_pos : np.ndarray
        2D position vector of the target [x, y].

    Returns
    -------
    float
        The straight-line distance in metres between the two points.
    """
    return float(np.linalg.norm(target_pos - missile_pos))


def telemetry_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Convert a list of telemetry records (dicts) into a pandas DataFrame.

    Parameters
    ----------
    records : list of dict
        Each dictionary should contain keys such as:
        't', 'range', 'los_rate', 'v_closing', etc.

    Returns
    -------
    pandas.DataFrame
        A tabular representation suitable for plotting or export.
    """
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# Example placeholder for future Monte Carlo integration
def estimate_hit_probability(results: list[bool]) -> float:
    """
    Estimate hit probability given a sequence of Boolean outcomes.

    Parameters
    ----------
    results : list of bool
        True for a hit, False for a miss in each trial.

    Returns
    -------
    float
        Hit probability in percentage (0.0–100.0).
    """
    if not results:
        return 0.0
    return 100.0 * sum(results) / len(results)
