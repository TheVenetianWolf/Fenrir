"""Telemetry utilities for FENRIR.

This module provides:
- A typed `TelemetryRecord` representing one snapshot of the simulation.
- A `TelemetryBuffer` container to accumulate per-step records with optional
  ring-buffer behaviour.
- Small math helpers for common engagement metrics (range, LOS rate, closing speed).

Conventions:
- 2D kinematics in metres [m] and seconds [s]
- Velocities [m/s], accelerations [m/s²], angles [rad]
- All vectors are NumPy arrays of shape (2,).

The API intentionally uses *duck typing* for entities: anything with
`.state.r` and `.state.v` (and optional telemetry fields) is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Iterable, Optional, Sequence

import numpy as np

try:  # pandas is optional at import time
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - avoid hard dep in engine-only contexts
    pd = None  # type: ignore


# ==============================================================================
# Metric helpers
# ==============================================================================

def los_metrics(r_m: np.ndarray, v_m: np.ndarray, r_t: np.ndarray, v_t: np.ndarray) -> tuple[float, float, float]:
    """Compute range, LOS rate (λ̇), and closing speed (Vc) in 2D.

    Args:
        r_m: Missile position [m], shape (2,).
        v_m: Missile velocity [m/s], shape (2,).
        r_t: Target position [m], shape (2,).
        v_t: Target velocity [m/s], shape (2,).

    Returns:
        A tuple of:
            range_m (float): Euclidean separation [m]
            los_rate (float): LOS angular rate λ̇ [rad/s] (z-component)
            v_closing (float): Closing speed Vc [m/s] (positive when closing)

    Notes:
        In 2D, λ̇ is computed as the z-component of cross(l̂, v_rel) / |r|,
        where l̂ = r / |r| and v_rel = v_t - v_m.
    """
    r = r_t - r_m
    v_rel = v_t - v_m
    rng = float(np.linalg.norm(r) + 1e-12)
    lhat = r / rng
    # cross in 2D via embedding in z
    los_rate = float(np.cross(np.append(lhat, 0.0), np.append(v_rel, 0.0))[2] / rng)
    v_closing = float(-np.dot(v_rel, lhat))
    return rng, los_rate, v_closing


# ==============================================================================
# Telemetry record & buffer
# ==============================================================================

@dataclass(slots=True)
class TelemetryRecord:
    """One snapshot of engagement telemetry.

    Fields mirror those used by the Streamlit UI for plotting/export.

    Attributes:
        t: Simulation time [s].
        range: Range between missile and target [m].
        los_rate: LOS angular rate λ̇ [rad/s].
        v_closing: Closing speed Vc [m/s].
        m_speed: Missile speed ||v_m|| [m/s].
        a_cmd: Commanded lateral acceleration [m/s²] (from guidance).
        a_lat: Signed lateral accel magnitude applied [m/s²] (before drag etc.).
        a_ach: Achieved total accel magnitude [m/s²] (dynamics result).
        mx, my: Missile position [m].
        tx, ty: Target position [m].
    """
    t: float
    range: float
    los_rate: float
    v_closing: float
    m_speed: float
    a_cmd: float
    a_lat: float
    a_ach: float
    mx: float
    my: float
    tx: float
    ty: float

    @classmethod
    def from_entities(cls, missile, target, t: float) -> "TelemetryRecord":
        """Construct a record from missile/target entities (duck-typed).

        The entities are expected to have:
            - `.state.r` (np.ndarray shape (2,))
            - `.state.v` (np.ndarray shape (2,))
        The missile may optionally have:
            - `.last_a_cmd`, `.last_a_lat`, `.last_a_achieved`

        Args:
            missile: Missile-like object with `state` (r, v) and optional telemetry attrs.
            target: Target-like object with `state` (r, v).
            t: Current simulation time [s].

        Returns:
            TelemetryRecord populated from current states.
        """
        r_m = np.asarray(missile.state.r, dtype=float)
        v_m = np.asarray(missile.state.v, dtype=float)
        r_t = np.asarray(target.state.r, dtype=float)
        v_t = np.asarray(target.state.v, dtype=float)

        rng, los_rate, v_closing = los_metrics(r_m, v_m, r_t, v_t)
        m_speed = float(np.linalg.norm(v_m))

        a_cmd = float(getattr(missile, "last_a_cmd", 0.0))
        a_lat = float(getattr(missile, "last_a_lat", 0.0))
        a_ach = float(getattr(missile, "last_a_achieved", 0.0))

        return cls(
            t=float(t),
            range=float(rng),
            los_rate=float(los_rate),
            v_closing=float(v_closing),
            m_speed=m_speed,
            a_cmd=a_cmd,
            a_lat=a_lat,
            a_ach=a_ach,
            mx=float(r_m[0]),
            my=float(r_m[1]),
            tx=float(r_t[0]),
            ty=float(r_t[1]),
        )

    def as_dict(self) -> dict:
        """Return a plain `dict` (useful for JSON/CSV export)."""
        return asdict(self)


class TelemetryBuffer:
    """A small container for telemetry snapshots with optional ring behaviour.

    Example:
        buf = TelemetryBuffer(capacity=10_000)
        for step in sim:
            buf.append(TelemetryRecord.from_entities(m, t, world.t))
        df = buf.to_dataframe()

    Args:
        capacity: If provided, buffer acts as a ring buffer keeping at most
            `capacity` most recent records. If `None`, it grows unbounded.

    Notes:
        - `to_dataframe` requires pandas; otherwise use `to_dicts`.
        - Thread-safety is not provided; Streamlit runs are single-threaded per session.
    """

    def __init__(self, capacity: Optional[int] = None) -> None:
        self._cap = int(capacity) if capacity is not None else None
        self._data: list[TelemetryRecord] = []

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def clear(self) -> None:
        """Remove all records."""
        self._data.clear()

    def append(self, rec: TelemetryRecord | dict) -> None:
        """Append a record, respecting capacity if ring-buffering is enabled.

        Args:
            rec: Either a `TelemetryRecord` or a dict with compatible keys.
        """
        if not isinstance(rec, TelemetryRecord):
            rec = TelemetryRecord(**rec)  # type: ignore[arg-type]
        self._data.append(rec)
        if self._cap is not None and len(self._data) > self._cap:
            # Drop oldest
            del self._data[0 : len(self._data) - self._cap]

    def extend(self, recs: Iterable[TelemetryRecord | dict]) -> None:
        """Append many records efficiently."""
        for r in recs:
            self.append(r)

    def to_dicts(self) -> list[dict]:
        """Return a list of dictionaries (stable for JSON/CSV)."""
        return [r.as_dict() for r in self._data]

    def to_dataframe(self):
        """Return a pandas DataFrame of the buffer (if pandas is available).

        Returns:
            pandas.DataFrame

        Raises:
            ImportError: If pandas is not installed.
        """
        if pd is None:
            raise ImportError("pandas is required for `TelemetryBuffer.to_dataframe()`")
        return pd.DataFrame(self.to_dicts())

    def to_csv(self, path: str, index: bool = False) -> None:
        """Write the buffer as CSV (requires pandas).

        Args:
            path: Output file path.
            index: Include a row index in the CSV.
        """
        df = self.to_dataframe()
        df.to_csv(path, index=index)
