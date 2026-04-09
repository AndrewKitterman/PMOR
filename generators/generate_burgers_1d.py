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
    periodic_series,
    save_dataset,
    stable_time_grid,
)


@dataclass
class Burgers1DConfig:
    output_dir: Path = Path("data/burgers_1d")
    seed: int = 14142
    num_points: int = 192
    train_num_snapshots: int = 32
    test_num_snapshots: int = 64
    train_trajectories: int = DEFAULT_TRAIN_TRAJECTORIES
    total_trajectories: int = DEFAULT_TOTAL_TRAJECTORIES
    train_history_fraction: float = 0.30
    test_history_fraction: float = 2.0
    ic_modes: int = 7
    forcing_decay: float = 1.25


def build_parameter_matrix(config: Burgers1DConfig) -> tuple[np.ndarray, list[str]]:
    physical_bounds = [
        ("viscosity", 0.016, 0.065),
        ("mean_level", -0.45, 0.45),
        ("initial_amplitude", 0.22, 0.95),
        ("forcing_amplitude", 0.00, 0.18),
    ]
    physical_parameters, parameter_names = parameter_table(
        physical_bounds,
        total_trajectories=config.total_trajectories,
        seed=config.seed,
    )
    ic_seeds = np.array(
        [deterministic_seed("burgers_1d", config.seed, index) for index in range(config.total_trajectories)],
        dtype=np.float64,
    ).reshape(-1, 1)
    return np.hstack([physical_parameters, ic_seeds]), parameter_names + ["initial_condition_seed"]


def burgers_rhs(values: np.ndarray, viscosity: float, dx: float) -> np.ndarray:
    flux = 0.5 * values * values
    right = np.roll(values, -1)
    left = np.roll(values, 1)
    wave_speed_right = np.maximum(np.abs(values), np.abs(right))
    wave_speed_left = np.maximum(np.abs(left), np.abs(values))

    flux_right = 0.5 * (flux + np.roll(flux, -1)) - 0.5 * wave_speed_right * (right - values)
    flux_left = 0.5 * (np.roll(flux, 1) + flux) - 0.5 * wave_speed_left * (values - left)
    advection = -(flux_right - flux_left) / dx
    diffusion = viscosity * (right - 2.0 * values + left) / (dx * dx)
    return advection + diffusion


def periodic_bump_field(x: np.ndarray, *, seed: int, num_bumps: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = np.zeros_like(x, dtype=np.float64)
    for _ in range(num_bumps):
        center = rng.uniform(0.0, 1.0)
        width = rng.uniform(0.035, 0.11)
        weight = rng.uniform(-1.0, 1.0)
        wrapped_distance = np.minimum(np.abs(x - center), 1.0 - np.abs(x - center))
        field += weight * np.exp(-0.5 * (wrapped_distance / width) ** 2)

    field -= field.mean()
    scale = np.max(np.abs(field))
    if scale < 1.0e-12:
        return np.zeros_like(x)
    return field / scale


def solve_trajectory(
    x: np.ndarray,
    parameters: np.ndarray,
    config: Burgers1DConfig,
    *,
    history_fraction: float,
    num_snapshots: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    viscosity, mean_level, initial_amplitude, forcing_amplitude, initial_condition_seed = parameters
    dx = float(x[1] - x[0])
    steady_time_estimate = 4.5 / (viscosity * (2.0 * np.pi) ** 2)
    total_time = history_fraction * steady_time_estimate
    speed_bound = max(1.8, abs(mean_level) + 1.6 * initial_amplitude + 0.5 * forcing_amplitude)
    max_dt = min(0.25 * dx / speed_bound, 0.18 * dx * dx / viscosity)
    times, save_indices, dt, num_steps = stable_time_grid(
        total_time=total_time,
        max_dt=max_dt,
        num_snapshots=num_snapshots,
    )

    seed_base = int(round(initial_condition_seed))
    state = mean_level + periodic_series(
        x,
        seed=seed_base,
        amplitude=float(initial_amplitude),
        modes=config.ic_modes,
    )
    state += 0.55 * initial_amplitude * periodic_bump_field(
        x,
        seed=deterministic_seed("burgers_bumps", seed_base),
    )
    state -= state.mean()
    state += mean_level
    forcing_profile = periodic_series(
        x,
        seed=deterministic_seed("burgers_forcing", seed_base),
        amplitude=1.0,
        modes=3,
    )
    forcing_profile += 0.4 * periodic_bump_field(
        x,
        seed=deterministic_seed("burgers_forcing_bumps", seed_base),
    )
    forcing_profile -= forcing_profile.mean()
    forcing_scale = np.max(np.abs(forcing_profile))
    if forcing_scale > 1.0e-12:
        forcing_profile /= forcing_scale
    snapshots: list[np.ndarray] = []
    save_pointer = 0
    if save_indices[save_pointer] == 0:
        snapshots.append(state.astype(np.float32))
        save_pointer += 1

    for step in range(1, num_steps + 1):
        time_start = (step - 1) * dt
        forcing_0 = forcing_amplitude * np.exp(-config.forcing_decay * time_start) * forcing_profile
        stage_1 = state + dt * (burgers_rhs(state, float(viscosity), dx) + forcing_0)

        time_mid = time_start + dt
        forcing_1 = forcing_amplitude * np.exp(-config.forcing_decay * time_mid) * forcing_profile
        stage_2 = 0.75 * state + 0.25 * (
            stage_1 + dt * (burgers_rhs(stage_1, float(viscosity), dx) + forcing_1)
        )

        time_end = time_start + 0.5 * dt
        forcing_2 = forcing_amplitude * np.exp(-config.forcing_decay * time_end) * forcing_profile
        next_state = (
            state
            + 2.0 * (stage_2 + dt * (burgers_rhs(stage_2, float(viscosity), dx) + forcing_2))
        ) / 3.0
        assert_finite("burgers_1d_state", next_state)
        state = next_state
        if save_pointer < save_indices.size and step == save_indices[save_pointer]:
            snapshots.append(state.astype(np.float32))
            save_pointer += 1

    state_history = np.stack(snapshots, axis=0)
    return state_history, times.astype(np.float32), steady_time_estimate


def generate_dataset(config: Burgers1DConfig | None = None) -> dict[str, Path]:
    config = config or Burgers1DConfig()
    x = np.linspace(0.0, 1.0, config.num_points, endpoint=False, dtype=np.float64)
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
        "case": "burgers_1d",
        "equation": "u_t + (0.5 * u^2)_x = viscosity * u_xx + forcing(x, t)",
        "boundary_conditions": "Periodic",
        "notes": "Each trajectory uses a sharper random periodic initial condition plus a deterministic zero-mean decaying forcing pulse, which yields richer shock-merging transients before the solution relaxes toward its constant mean state.",
        "config": asdict(config),
    }
    train_path = config.output_dir / "train.npz"
    test_path = config.output_dir / "test.npz"
    save_dataset(
        train_path,
        case_name="burgers_1d",
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
        case_name="burgers_1d",
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


def parse_args() -> Burgers1DConfig:
    parser = argparse.ArgumentParser(description="Generate the 1D Burgers PMOR dataset.")
    parser.add_argument("--output-dir", type=Path, default=Burgers1DConfig.output_dir)
    parser.add_argument("--seed", type=int, default=Burgers1DConfig.seed)
    parser.add_argument("--num-points", type=int, default=Burgers1DConfig.num_points)
    parser.add_argument("--train-num-snapshots", type=int, default=Burgers1DConfig.train_num_snapshots)
    parser.add_argument("--test-num-snapshots", type=int, default=Burgers1DConfig.test_num_snapshots)
    parser.add_argument("--train-trajectories", type=int, default=Burgers1DConfig.train_trajectories)
    parser.add_argument("--total-trajectories", type=int, default=Burgers1DConfig.total_trajectories)
    parser.add_argument("--train-history-fraction", type=float, default=Burgers1DConfig.train_history_fraction)
    parser.add_argument("--test-history-fraction", type=float, default=Burgers1DConfig.test_history_fraction)
    parser.add_argument("--ic-modes", type=int, default=Burgers1DConfig.ic_modes)
    parser.add_argument("--forcing-decay", type=float, default=Burgers1DConfig.forcing_decay)
    args = parser.parse_args()
    return Burgers1DConfig(
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
        forcing_decay=args.forcing_decay,
    )


def main() -> None:
    generate_dataset(parse_args())


if __name__ == "__main__":
    main()
