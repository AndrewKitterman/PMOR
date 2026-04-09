from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from generators.common import (
    DEFAULT_TOTAL_TRAJECTORIES,
    DEFAULT_TRAIN_TRAJECTORIES,
    assert_finite,
    deterministic_seed,
    parameter_table,
    save_dataset,
    sine_series,
    stable_time_grid,
)


@dataclass
class DampedWave1DConfig:
    output_dir: Path = Path("data/damped_wave_1d")
    seed: int = 27182
    num_points: int = 129
    train_num_snapshots: int = 32
    test_num_snapshots: int = 64
    train_trajectories: int = DEFAULT_TRAIN_TRAJECTORIES
    total_trajectories: int = DEFAULT_TOTAL_TRAJECTORIES
    train_history_fraction: float = 0.25
    test_history_fraction: float = 2.0
    ic_modes: int = 5


def build_parameter_matrix(config: DampedWave1DConfig) -> tuple[np.ndarray, list[str]]:
    physical_bounds = [
        ("wave_speed", 0.8, 1.8),
        ("damping", 0.6, 1.4),
        ("left_boundary", -0.25, 0.25),
        ("right_boundary", -0.25, 0.25),
        ("displacement_amplitude", 0.06, 0.32),
        ("velocity_amplitude", 0.04, 0.20),
    ]
    physical_parameters, parameter_names = parameter_table(
        physical_bounds,
        total_trajectories=config.total_trajectories,
        seed=config.seed,
    )
    ic_seeds = np.array(
        [deterministic_seed("damped_wave_1d", config.seed, index) for index in range(config.total_trajectories)],
        dtype=np.float64,
    ).reshape(-1, 1)
    return np.hstack([physical_parameters, ic_seeds]), parameter_names + ["initial_condition_seed"]


def laplacian_1d(values: np.ndarray, dx: float) -> np.ndarray:
    lap = np.zeros_like(values, dtype=np.float64)
    lap[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / (dx * dx)
    return lap


def solve_trajectory(
    x: np.ndarray,
    parameters: np.ndarray,
    config: DampedWave1DConfig,
    *,
    history_fraction: float,
    num_snapshots: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    (
        wave_speed,
        damping,
        left_bc,
        right_bc,
        displacement_amplitude,
        velocity_amplitude,
        initial_condition_seed,
    ) = parameters
    dx = float(x[1] - x[0])
    steady_time_estimate = 4.0 / damping
    total_time = history_fraction * steady_time_estimate
    max_dt = 0.80 * dx / wave_speed
    times, save_indices, dt, num_steps = stable_time_grid(
        total_time=total_time,
        max_dt=max_dt,
        num_snapshots=num_snapshots,
    )

    steady_profile = left_bc + (right_bc - left_bc) * x
    seed_base = int(round(initial_condition_seed))
    displacement = sine_series(
        x,
        seed=seed_base,
        amplitude=float(displacement_amplitude),
        modes=config.ic_modes,
    )
    velocity = sine_series(
        x,
        seed=deterministic_seed("velocity", seed_base),
        amplitude=float(velocity_amplitude),
        modes=config.ic_modes,
    )
    displacement[0] = 0.0
    displacement[-1] = 0.0
    velocity[0] = 0.0
    velocity[-1] = 0.0

    snapshots: list[np.ndarray] = []
    save_pointer = 0
    if save_indices[save_pointer] == 0:
        snapshots.append((displacement + steady_profile).astype(np.float32))
        save_pointer += 1

    acceleration = wave_speed**2 * laplacian_1d(displacement, dx) - damping * velocity
    current = displacement + dt * velocity + 0.5 * dt * dt * acceleration
    current[0] = 0.0
    current[-1] = 0.0
    previous = displacement

    if num_steps >= 1:
        assert_finite("damped_wave_1d_state", current)
        if save_pointer < save_indices.size and save_indices[save_pointer] == 1:
            snapshots.append((current + steady_profile).astype(np.float32))
            save_pointer += 1

    for step in range(1, num_steps):
        lap = laplacian_1d(current, dx)
        next_state = (
            2.0 * current
            - (1.0 - 0.5 * damping * dt) * previous
            + wave_speed**2 * dt * dt * lap
        ) / (1.0 + 0.5 * damping * dt)
        next_state[0] = 0.0
        next_state[-1] = 0.0
        assert_finite("damped_wave_1d_state", next_state)
        previous, current = current, next_state
        current_step = step + 1
        if save_pointer < save_indices.size and current_step == save_indices[save_pointer]:
            snapshots.append((current + steady_profile).astype(np.float32))
            save_pointer += 1

    state_history = np.stack(snapshots, axis=0)
    return state_history, times.astype(np.float32), steady_time_estimate


def generate_dataset(config: DampedWave1DConfig | None = None) -> dict[str, Path]:
    config = config or DampedWave1DConfig()
    x = np.linspace(0.0, 1.0, config.num_points, dtype=np.float64)
    parameter_matrix, parameter_names = build_parameter_matrix(config)

    train_states: list[np.ndarray] = []
    train_times: list[np.ndarray] = []
    test_states: list[np.ndarray] = []
    test_times: list[np.ndarray] = []
    steady_time_estimates = np.zeros(config.total_trajectories, dtype=np.float64)

    for trajectory_index, parameters in enumerate(parameter_matrix):
        if trajectory_index < config.train_trajectories:
            states, times, steady_time = solve_trajectory(
                x,
                parameters,
                config,
                history_fraction=config.train_history_fraction,
                num_snapshots=config.train_num_snapshots,
            )
            assert states.shape[0] == config.train_num_snapshots
            train_states.append(states)
            train_times.append(times)
        else:
            states, times, steady_time = solve_trajectory(
                x,
                parameters,
                config,
                history_fraction=config.test_history_fraction,
                num_snapshots=config.test_num_snapshots,
            )
            assert states.shape[0] == config.test_num_snapshots
            test_states.append(states)
            test_times.append(times)
        steady_time_estimates[trajectory_index] = steady_time

    metadata = {
        "case": "damped_wave_1d",
        "equation": "u_tt + damping * u_t = wave_speed^2 * u_xx",
        "boundary_conditions": "Dirichlet with parameterized left/right displacements",
        "notes": "Training trajectories cover the early transient, while the held-out test trajectory runs through the long-time damped tail toward the static profile.",
        "config": asdict(config),
    }
    output_dir = config.output_dir
    train_path = output_dir / "train.npz"
    test_path = output_dir / "test.npz"
    save_dataset(
        train_path,
        case_name="damped_wave_1d",
        split_name="train",
        state_sequences=train_states,
        time_sequences=train_times,
        parameter_matrix=parameter_matrix[: config.train_trajectories],
        parameter_names=parameter_names,
        steady_time_estimates=steady_time_estimates[: config.train_trajectories],
        grid_data={"x": x},
        metadata=metadata,
    )
    save_dataset(
        test_path,
        case_name="damped_wave_1d",
        split_name="test",
        state_sequences=test_states,
        time_sequences=test_times,
        parameter_matrix=parameter_matrix[config.train_trajectories :],
        parameter_names=parameter_names,
        steady_time_estimates=steady_time_estimates[config.train_trajectories :],
        grid_data={"x": x},
        metadata=metadata,
    )
    return {"train": train_path, "test": test_path}


def parse_args() -> DampedWave1DConfig:
    parser = argparse.ArgumentParser(description="Generate the damped-wave 1D PMOR dataset.")
    parser.add_argument("--output-dir", type=Path, default=DampedWave1DConfig.output_dir)
    parser.add_argument("--seed", type=int, default=DampedWave1DConfig.seed)
    parser.add_argument("--num-points", type=int, default=DampedWave1DConfig.num_points)
    parser.add_argument("--train-num-snapshots", type=int, default=DampedWave1DConfig.train_num_snapshots)
    parser.add_argument("--test-num-snapshots", type=int, default=DampedWave1DConfig.test_num_snapshots)
    parser.add_argument("--train-trajectories", type=int, default=DampedWave1DConfig.train_trajectories)
    parser.add_argument("--total-trajectories", type=int, default=DampedWave1DConfig.total_trajectories)
    parser.add_argument("--train-history-fraction", type=float, default=DampedWave1DConfig.train_history_fraction)
    parser.add_argument("--test-history-fraction", type=float, default=DampedWave1DConfig.test_history_fraction)
    parser.add_argument("--ic-modes", type=int, default=DampedWave1DConfig.ic_modes)
    args = parser.parse_args()
    return DampedWave1DConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        num_points=args.num_points,
        train_num_snapshots=args.train_num_snapshots,
        test_num_snapshots=args.test_num_snapshots,
        train_trajectories=args.train_trajectories,
        total_trajectories=args.total_trajectories,
        train_history_fraction=args.train_history_fraction,
        test_history_fraction=args.test_history_fraction,
        ic_modes=args.ic_modes,
    )


def main() -> None:
    generate_dataset(parse_args())


if __name__ == "__main__":
    main()
