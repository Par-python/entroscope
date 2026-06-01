"""Synthetic-data generators for the entroscope examples.

These produce realistic-shaped signals so every example runs with no external
data files. Each generator returns a ``pd.Series`` with a meaningful index.
Real usage would swap these for ``pd.read_csv(...)`` of your own data.
"""

import numpy as np
import pandas as pd


def _rng(seed):
    return np.random.RandomState(seed)


# ---------------------------------------------------------------- medical ---


def heart_rate(seed=0):
    """Heart-rate signal: healthy variable HR, then an abnormally regular run.

    Reduced beat-to-beat variability (a more regular signal) is clinically
    meaningful — sample entropy drops when it happens.
    """
    rng = _rng(seed)
    healthy = 70 + 6 * rng.randn(300)  # high variability
    regular = 70 + 0.8 * np.sin(np.linspace(0, 30, 300)) + 0.4 * rng.randn(300)
    values = np.concatenate([healthy, regular])
    idx = pd.date_range("2025-01-01", periods=len(values), freq="s")
    return pd.Series(values, index=idx, name="bpm")


def eeg(seed=1):
    """EEG-like trace: desynchronized background, then a rhythmic seizure burst.

    A seizure shows up as the signal collapsing onto a dominant rhythm, which
    drives permutation and spectral entropy down.
    """
    rng = _rng(seed)
    t = np.linspace(0, 12, 600)
    background = rng.randn(600)  # broadband, no structure
    seizure = 4 * np.sin(2 * np.pi * 3 * t) + 0.3 * rng.randn(600)  # 3 Hz rhythm
    values = np.concatenate([background, seizure])
    return pd.Series(values, name="eeg_uv")


def respiration(seed=2):
    """Respiration trace: steady breathing, then irregular/labored breathing.

    Approximate entropy rises as the regular cycle breaks down.
    """
    rng = _rng(seed)
    t = np.linspace(0, 40, 400)
    steady = np.sin(t) + 0.02 * rng.randn(400)  # clean periodic cycle
    # irregular phase: wandering rate, varying depth, and added noise
    jitter = np.cumsum(0.15 * rng.randn(400))  # drifting phase
    depth = 1 + 0.5 * rng.randn(400)  # uneven breath depth
    irregular = depth * np.sin(t + jitter) + 0.3 * rng.randn(400)
    values = np.concatenate([steady, irregular])
    idx = pd.date_range("2025-01-01", periods=len(values), freq="500ms")
    return pd.Series(values, index=idx, name="chest_expansion")


def glucose(seed=3):
    """Continuous glucose monitor: stable control, then a volatile regime.

    Differential entropy (a function of spread) climbs when variability rises.
    """
    rng = _rng(seed)
    stable = 110 + 5 * rng.randn(240)  # tight, well-controlled
    volatile = 140 + 50 * rng.randn(240)  # wide erratic swings
    values = np.clip(np.concatenate([stable, volatile]), 50, 300)
    idx = pd.date_range("2025-01-01", periods=len(values), freq="5min")
    return pd.Series(values, index=idx, name="glucose_mgdl")


# --------------------------------------------------------------- business ---


def daily_sales(seed=10):
    """Daily sales: erratic launch period, then a settled weekly pattern.

    Rolling Shannon entropy falls as demand consolidates into a stable shape.
    """
    rng = _rng(seed)
    erratic = rng.uniform(100, 1500, 90)  # launch: spread everywhere
    week = np.array([1.0, 1.05, 1.0, 1.1, 1.3, 0.55, 0.4])
    settled = 800 * np.tile(week, 13)[:90] + 15 * rng.randn(90)  # tight pattern
    values = np.concatenate([erratic, settled])
    idx = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, name="units_sold")


def web_traffic(seed=11):
    """Hourly request volume: clean daily cycle, then a bot-driven anomaly.

    Spectral entropy is low while a single daily cycle dominates and jumps when
    irregular bursts spread power across many frequencies.
    """
    rng = _rng(seed)
    t = np.arange(480)
    daily = 1000 + 600 * np.sin(2 * np.pi * t / 24) + 50 * rng.randn(480)
    bots = daily.copy()
    bots[240:] += 800 * rng.rand(240) * (rng.rand(240) > 0.6)  # sporadic spikes
    idx = pd.date_range("2025-01-01", periods=len(bots), freq="h")
    return pd.Series(np.clip(bots, 0, None), index=idx, name="requests")


def stock_prices(seed=12):
    """Daily close prices: calm trending market, then a turbulent regime.

    Permutation entropy is a noise-robust gauge of how disordered price moves
    are — it rises when the market becomes erratic.
    """
    rng = _rng(seed)
    calm = np.cumsum(0.2 + 0.5 * rng.randn(150)) + 100  # gentle drift
    turbulent = calm[-1] + np.cumsum(2.5 * rng.randn(150))  # wild swings
    values = np.concatenate([calm, turbulent])
    idx = pd.date_range("2025-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="close")


def qc_sensor(seed=13):
    """Manufacturing sensor stream: in-control process, then drift + chatter.

    Multiscale entropy reveals complexity changes across time scales as the
    process leaves its controlled state.
    """
    rng = _rng(seed)
    in_control = 50 + 0.5 * rng.randn(400)
    out_of_control = 50 + np.linspace(0, 3, 400) + 1.5 * rng.randn(400)
    values = np.concatenate([in_control, out_of_control])
    return pd.Series(values, name="dimension_mm")
