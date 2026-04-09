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
    disk_mask,
    gaussian_blob_field,
    parameter_table,
    save_dataset,
    stable_time_grid,
)


@dataclass
class FisherKPPDiskConfig:
    output_dir: Path = Path("data/fisher_kpp_disk")
    seed: int = 17320
    grid_size: int = 48
    num_snapshots: int = 24
    train_trajectories: int = DEFAULT_TRAIN_TRAJECTORIES
    total_trajectories: int = DEFAULT_TOTAL_TRAJECTORIES
    history_fraction: float = 0.25


def build_parameter_matrix(config: FisherKPPDiskConfig) -> tuple[np.ndarray, list[str]]:
    physical_bounds = [
        ("diffusion", 0.010, 0.040),
        ("growth_rate", 0.70, 1.60),
        ("initial_amplitude", 0.35, 0.95),
    ]
    physical_parameters, parameter_names = parameter_table(
        physical_bounds,
        total_trajectories=config.total_trajectories,
        seed=config.seed,
    )
    ic_seeds = np.array(
        [deterministic_seed("fisher_kpp_disk", config.seed, index) for index in range(config.total_trajectories)],
        dtype=np.float64,
    ).reshape(-1, 1)
    return np.hstack([physical_parameters, ic_seeds]), parameter_names + ["initial_condition_seed"]


def solve_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    parameters: np.ndarray,
    mask: np.ndarray,
    config: FisherKPPDiskConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    diffusion, growth_rate, initial_amplitude, initial_condition_seed = parameters
    dx = float(x[1] - x[0])
    effective_rate = growth_rate + 10.0 * diffusion
    steady_time_estimate = 4.0 / effective_rate
    total_time = config.history_fraction * steady_time_estimate
    max_dt = 0.22 * dx * dx / diffusion
    times, save_indices, dt, num_steps = stable_time_grid(
        total_time=total_time,
        max_dt=max_dt,
        num_snapshots=config.num_snapshots,
    )

    xx, yy = np.meshgrid(x, y, indexing="xy")
    state = float(initial_amplitude) * gaussian_blob_field(
        xx,
        yy,
        mask,
        seed=int(round(initial_condition_seed)),
    )
    state *= mask

    snapshots: list[np.ndarray] = []
    save_pointer = 0
    if save_indices[save_pointer] == 0:
        snapshots.append(state.astype(np.float32))
        save_pointer += 1

    for step in range(1, num_steps + 1):
        padded = np.pad(state, 1, mode="constant", constant_values=0.0)
        laplacian = (
            padded[1:-1, 2:]
            + padded[1:-1, :-2]
            + padded[2:, 1:-1]
            + padded[:-2, 1:-1]
            - 4.0 * state
        ) / (dx * dx)
        diffused = state + diffusion * dt * laplacian
        diffused *= mask

        growth_factor = np.exp(growth_rate * dt)
        next_state = diffused * growth_factor / (1.0 + diffused * (growth_factor - 1.0))
        next_state = np.clip(next_state, 0.0, 1.25)
        next_state *= mask
        assert_finite("fisher_kpp_disk_state", next_state)
        state = next_state
        if save_pointer < save_indices.size and step == save_indices[save_pointer]:
            snapshots.append(state.astype(np.float32))
            save_pointer += 1

    state_history = np.stack(snapshots, axis=0)
    return state_history, times.astype(np.float32), steady_time_estimate


def generate_dataset(config: FisherKPPDiskConfig | None = None) -> dict[str, Path]:
    config = config or FisherKPPDiskConfig()
    x = np.linspace(0.0, 1.0, config.grid_size, dtype=np.float64)
    y = np.linspace(0.0, 1.0, config.grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mask = disk_mask(xx, yy, center=(0.5, 0.5), radius=0.45)
    parameter_matrix, parameter_names = build_parameter_matrix(config)

    train_states: list[np.ndarray] = []
    train_times: list[np.ndarray] = []
    test_states: list[np.ndarray] = []
    test_times: list[np.ndarray] = []
    steady_time_estimates = np.zeros(config.total_trajectories, dtype=np.float64)

    for trajectory_index, parameters in enumerate(parameter_matrix):
        states, times, steady_time = solve_trajectory(x, y, parameters, mask, config)
        assert states.shape[0] == config.num_snapshots
        steady_time_estimates[trajectory_index] = steady_time
        if trajectory_index < config.train_trajectories:
            train_states.append(states)
            train_times.append(times)
        else:
            test_states.append(states)
            test_times.append(times)

    metadata = {
        "case": "fisher_kpp_disk",
        "equation": "u_t = diffusion * Laplacian(u) + growth_rate * u * (1 - u)",
        "boundary_conditions": "Embedded disk with homogeneous Dirichlet boundary values",
        "notes": "This is the nonlinear 2D disk geometry case.",
        "config": asdict(config),
    }
    train_path = config.output_dir / "train.npz"
    test_path = config.output_dir / "test.npz"
    save_dataset(
        train_path,
        case_name="fisher_kpp_disk",
        split_name="train",
        state_sequences=train_states,
        time_sequences=train_times,
        parameter_matrix=parameter_matrix[: config.train_trajectories],
        parameter_names=parameter_names,
        steady_time_estimates=steady_time_estimates[: config.train_trajectories],
        grid_data={"x": x, "y": y, "mask": mask.astype(np.int8)},
        metadata=metadata,
    )
    save_dataset(
        test_path,
        case_name="fisher_kpp_disk",
        split_name="test",
        state_sequences=test_states,
        time_sequences=test_times,
        parameter_matrix=parameter_matrix[config.train_trajectories :],
        parameter_names=parameter_names,
        steady_time_estimates=steady_time_estimates[config.train_trajectories :],
        grid_data={"x": x, "y": y, "mask": mask.astype(np.int8)},
        metadata=metadata,
    )
    return {"train": train_path, "test": test_path}


def parse_args() -> FisherKPPDiskConfig:
    parser = argparse.ArgumentParser(description="Generate the 2D Fisher-KPP disk PMOR dataset.")
    parser.add_argument("--output-dir", type=Path, default=FisherKPPDiskConfig.output_dir)
    parser.add_argument("--seed", type=int, default=FisherKPPDiskConfig.seed)
    parser.add_argument("--grid-size", type=int, default=FisherKPPDiskConfig.grid_size)
    parser.add_argument("--num-snapshots", type=int, default=FisherKPPDiskConfig.num_snapshots)
    parser.add_argument("--train-trajectories", type=int, default=FisherKPPDiskConfig.train_trajectories)
    parser.add_argument("--total-trajectories", type=int, default=FisherKPPDiskConfig.total_trajectories)
    parser.add_argument("--history-fraction", type=float, default=FisherKPPDiskConfig.history_fraction)
    args = parser.parse_args()
    return FisherKPPDiskConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        grid_size=args.grid_size,
        num_snapshots=args.num_snapshots,
        train_trajectories=args.train_trajectories,
        total_trajectories=args.total_trajectories,
        history_fraction=args.history_fraction,
    )


def main() -> None:
    generate_dataset(parse_args())


if __name__ == "__main__":
    main()
