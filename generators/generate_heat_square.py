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
    stable_time_grid,
)


@dataclass
class HeatSquareConfig:
    output_dir: Path = Path("data/heat_square")
    seed: int = 16180
    grid_size: int = 48
    train_num_snapshots: int = 24
    test_num_snapshots: int = 64
    train_trajectories: int = DEFAULT_TRAIN_TRAJECTORIES
    total_trajectories: int = DEFAULT_TOTAL_TRAJECTORIES
    train_history_fraction: float = 0.30
    test_history_fraction: float = 2.0
    ic_modes: int = 5
    source_decay: float = 1.40


def build_parameter_matrix(config: HeatSquareConfig) -> tuple[np.ndarray, list[str]]:
    physical_bounds = [
        ("diffusivity", 0.08, 0.24),
        ("left_boundary", -0.65, 0.65),
        ("right_boundary", -0.65, 0.65),
        ("bottom_boundary", -0.65, 0.65),
        ("top_boundary", -0.65, 0.65),
        ("initial_amplitude", 0.08, 0.35),
        ("source_amplitude", 0.00, 0.16),
    ]
    physical_parameters, parameter_names = parameter_table(
        physical_bounds,
        total_trajectories=config.total_trajectories,
        seed=config.seed,
    )
    ic_seeds = np.array(
        [deterministic_seed("heat_square", config.seed, index) for index in range(config.total_trajectories)],
        dtype=np.float64,
    ).reshape(-1, 1)
    return np.hstack([physical_parameters, ic_seeds]), parameter_names + ["initial_condition_seed"]


def interior_perturbation(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    seed: int,
    amplitude: float,
    modes: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = np.zeros_like(xx, dtype=np.float64)
    for mode_x in range(1, modes + 1):
        for mode_y in range(1, modes + 1):
            coeff = rng.uniform(-1.0, 1.0)
            field += coeff * np.sin(mode_x * np.pi * xx) * np.sin(mode_y * np.pi * yy)

    norm = np.max(np.abs(field))
    if norm < 1.0e-12:
        return np.zeros_like(field)
    return amplitude * field / norm


def signed_blob_field(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    seed: int,
    num_blobs: int = 3,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = np.zeros_like(xx, dtype=np.float64)
    for _ in range(num_blobs):
        center_x = rng.uniform(0.18, 0.82)
        center_y = rng.uniform(0.18, 0.82)
        width = rng.uniform(0.08, 0.20)
        weight = rng.uniform(-1.0, 1.0)
        field += weight * np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * width**2))

    field -= field.mean()
    norm = np.max(np.abs(field))
    if norm < 1.0e-12:
        return np.zeros_like(field)
    return field / norm


def apply_boundaries(
    values: np.ndarray,
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
) -> None:
    values[:, 0] = left
    values[:, -1] = right
    values[0, :] = bottom
    values[-1, :] = top
    values[0, 0] = 0.5 * (left + bottom)
    values[0, -1] = 0.5 * (right + bottom)
    values[-1, 0] = 0.5 * (left + top)
    values[-1, -1] = 0.5 * (right + top)


def solve_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    parameters: np.ndarray,
    config: HeatSquareConfig,
    *,
    history_fraction: float,
    num_snapshots: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    (
        diffusivity,
        left_boundary,
        right_boundary,
        bottom_boundary,
        top_boundary,
        initial_amplitude,
        source_amplitude,
        initial_condition_seed,
    ) = parameters
    dx = float(x[1] - x[0])
    steady_time_estimate = 2.5 / (diffusivity * np.pi**2)
    total_time = history_fraction * steady_time_estimate
    max_dt = 0.18 * dx * dx / diffusivity
    times, save_indices, dt, num_steps = stable_time_grid(
        total_time=total_time,
        max_dt=max_dt,
        num_snapshots=num_snapshots,
    )

    xx, yy = np.meshgrid(x, y, indexing="xy")
    boundary_average = 0.25 * (left_boundary + right_boundary + bottom_boundary + top_boundary)
    state = boundary_average + interior_perturbation(
        xx,
        yy,
        seed=int(round(initial_condition_seed)),
        amplitude=float(initial_amplitude),
        modes=config.ic_modes,
    )
    state += 0.45 * initial_amplitude * signed_blob_field(
        xx,
        yy,
        seed=deterministic_seed("heat_blobs", int(round(initial_condition_seed))),
    )
    apply_boundaries(
        state,
        left=float(left_boundary),
        right=float(right_boundary),
        bottom=float(bottom_boundary),
        top=float(top_boundary),
    )
    source_profile = signed_blob_field(
        xx,
        yy,
        seed=deterministic_seed("heat_source", int(round(initial_condition_seed))),
    )

    snapshots: list[np.ndarray] = []
    save_pointer = 0
    if save_indices[save_pointer] == 0:
        snapshots.append(state.astype(np.float32))
        save_pointer += 1

    for step in range(1, num_steps + 1):
        current_time = (step - 1) * dt
        source = source_amplitude * np.exp(-config.source_decay * current_time) * source_profile
        laplacian = (
            state[1:-1, 2:]
            + state[1:-1, :-2]
            + state[2:, 1:-1]
            + state[:-2, 1:-1]
            - 4.0 * state[1:-1, 1:-1]
        ) / (dx * dx)
        updated = state.copy()
        updated[1:-1, 1:-1] = (
            state[1:-1, 1:-1]
            + diffusivity * dt * laplacian
            + dt * source[1:-1, 1:-1]
        )
        apply_boundaries(
            updated,
            left=float(left_boundary),
            right=float(right_boundary),
            bottom=float(bottom_boundary),
            top=float(top_boundary),
        )
        state = updated
        assert_finite("heat_square_state", state)
        if save_pointer < save_indices.size and step == save_indices[save_pointer]:
            snapshots.append(state.astype(np.float32))
            save_pointer += 1

    state_history = np.stack(snapshots, axis=0)
    return state_history, times.astype(np.float32), steady_time_estimate


def generate_dataset(config: HeatSquareConfig | None = None) -> dict[str, Path]:
    config = config or HeatSquareConfig()
    x = np.linspace(0.0, 1.0, config.grid_size, dtype=np.float64)
    y = np.linspace(0.0, 1.0, config.grid_size, dtype=np.float64)
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
                y,
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
                y,
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
        "case": "heat_square",
        "equation": "u_t = diffusivity * (u_xx + u_yy) + source(x, y, t)",
        "boundary_conditions": "Dirichlet on all four sides with parameterized values",
        "notes": "The square case now combines stronger off-center initial structure with a deterministic decaying interior heater-cooler field, so the transient develops moving thermal layers before settling back toward the boundary-driven steady solution.",
        "config": asdict(config),
    }
    train_path = config.output_dir / "train.npz"
    test_path = config.output_dir / "test.npz"
    save_dataset(
        train_path,
        case_name="heat_square",
        split_name="train",
        state_sequences=train_states,
        time_sequences=train_times,
        parameter_matrix=parameter_matrix[: config.train_trajectories],
        parameter_names=parameter_names,
        steady_time_estimates=steady_time_estimates[: config.train_trajectories],
        grid_data={"x": x, "y": y},
        metadata=metadata,
    )
    save_dataset(
        test_path,
        case_name="heat_square",
        split_name="test",
        state_sequences=test_states,
        time_sequences=test_times,
        parameter_matrix=parameter_matrix[config.train_trajectories :],
        parameter_names=parameter_names,
        steady_time_estimates=steady_time_estimates[config.train_trajectories :],
        grid_data={"x": x, "y": y},
        metadata=metadata,
    )
    return {"train": train_path, "test": test_path}


def parse_args() -> HeatSquareConfig:
    parser = argparse.ArgumentParser(description="Generate the 2D heat-on-square PMOR dataset.")
    parser.add_argument("--output-dir", type=Path, default=HeatSquareConfig.output_dir)
    parser.add_argument("--seed", type=int, default=HeatSquareConfig.seed)
    parser.add_argument("--grid-size", type=int, default=HeatSquareConfig.grid_size)
    parser.add_argument("--train-num-snapshots", type=int, default=HeatSquareConfig.train_num_snapshots)
    parser.add_argument("--test-num-snapshots", type=int, default=HeatSquareConfig.test_num_snapshots)
    parser.add_argument("--train-trajectories", type=int, default=HeatSquareConfig.train_trajectories)
    parser.add_argument("--total-trajectories", type=int, default=HeatSquareConfig.total_trajectories)
    parser.add_argument("--train-history-fraction", type=float, default=HeatSquareConfig.train_history_fraction)
    parser.add_argument("--test-history-fraction", type=float, default=HeatSquareConfig.test_history_fraction)
    parser.add_argument("--ic-modes", type=int, default=HeatSquareConfig.ic_modes)
    parser.add_argument("--source-decay", type=float, default=HeatSquareConfig.source_decay)
    args = parser.parse_args()
    return HeatSquareConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        grid_size=args.grid_size,
        train_num_snapshots=args.train_num_snapshots,
        test_num_snapshots=args.test_num_snapshots,
        train_trajectories=args.train_trajectories,
        total_trajectories=args.total_trajectories,
        train_history_fraction=args.train_history_fraction,
        test_history_fraction=args.test_history_fraction,
        ic_modes=args.ic_modes,
        source_decay=args.source_decay,
    )


def main() -> None:
    generate_dataset(parse_args())


if __name__ == "__main__":
    main()
