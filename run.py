"""FENRIR: lightweight CLI runner for the physics engine.

This module runs a headless simulation loop (no Streamlit UI) so you can
smoke-test the core engine, guidance, and dynamics from the terminal.

Conventions:
- 2D kinematics in metres [m] and seconds [s]
- Velocities [m/s], accelerations [m/s²], angles [rad]

Example:
    $ python run.py
    # or with custom params:
    $ python run.py --dt 0.02 --duration 45 --log-every 0.5 --throttle 0.0
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Callable

from services.scenarios import simple_pn_intercept  # default scenario factory

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger("fenrir.run")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the headless sim runner."""
    p = argparse.ArgumentParser(description="Run FENRIR headless simulation loop.")
    p.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Integration time step [s]. Default: 0.05",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Total simulated time [s]. Default: 60",
    )
    p.add_argument(
        "--log-every",
        type=float,
        default=0.5,
        help="Log state every N seconds of simulated time. Default: 0.5",
    )
    p.add_argument(
        "--throttle",
        type=float,
        default=0.0,
        help="Real-time sleep between steps [s] (0 for fastest). Default: 0",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default="pn",
        choices=["pn"],  # extend when you add more factories
        help="Which scenario to run. Default: pn",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO",
    )
    return p.parse_args()


# ------------------------------------------------------------------------------
# Scenario registry (extend as you add more)
# ------------------------------------------------------------------------------
_SCENARIOS: dict[str, Callable[[], object]] = {
    "pn": simple_pn_intercept,
    # "evasive": pn_vs_evasive,  # add when implemented
}


# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------
def main() -> None:
    """Run the headless simulation with periodic logging.

    Creates a world from the selected scenario, integrates for the requested
    duration, and logs missile/target positions at a fixed simulated cadence.
    """
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    # Build world from scenario factory
    try:
        factory = _SCENARIOS[args.scenario]
    except KeyError as exc:
        raise SystemExit(f"Unknown scenario '{args.scenario}'. Available: {list(_SCENARIOS)}") from exc

    world = factory()
    dt = float(args.dt)
    if dt <= 0.0:
        raise SystemExit("dt must be > 0")

    steps_total = int(max(0.0, args.duration) / dt)
    log_interval_steps = max(1, int(args.log_every / dt))

    logger.info(
        f"Starting FENRIR headless run  |  scenario={args.scenario}  dt={dt:.3f}s  "
        f"duration={args.duration:.1f}s  steps={steps_total}"
    )

    try:
        for i in range(steps_total):
            world.step(dt)

            # Periodic log (sim-time, missile/target positions)
            if i % log_interval_steps == 0:
                missile = world.entities[0]
                target = world.entities[1]
                logger.info(
                    "t=%6.2f  missile=[%8.2f %8.2f]  target=[%8.2f %8.2f]",
                    world.t,
                    float(missile.state.r[0]),
                    float(missile.state.r[1]),
                    float(target.state.r[0]),
                    float(target.state.r[1]),
                )

            # Optional real-time throttle (useful for debugging)
            if args.throttle > 0.0:
                time.sleep(args.throttle)

    except KeyboardInterrupt:
        logger.info("Interrupted by user at t=%.2f s", world.t)

    logger.info("Run complete at t=%.2f s", world.t)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
