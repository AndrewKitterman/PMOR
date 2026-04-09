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
    train_num_snapshots: int = 24
    test_num_snapshots: int = 64
    train_trajectories: int = DEFAULT_TRAIN_TRAJECTORIES
    total_trajectories: int = DEFAULT_TOTAL_TRAJECTORIES
    train_history_fraction: float = 0.30
    test_history_fraction: float = 2.0


def build_parameter_matrix(config: FisherKPPDiskConfig) -> tuple[np.ndarray, list[str]]:
    physical_bounds = [
        ("diffusion", 0.008, 0.028),
        ("growth_rate", 0.80, 1.90),
        ("initial_amplitude", 0.10, 0.75),
        ("carrying_capacity_contrast", 0.00, 0.40),
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


def signed_disk_field(
    xx: np.ndarray,
    yy: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
    num_blobs: int = 4,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = np.zeros_like(xx, dtype=np.float64)
    for _ in range(num_blobs):
        radius = np.sqrt(rng.uniform(0.0, 1.0)) * 0.30
        angle = rng.uniform(0.0, 2.0 * np.pi)
        center_x = 0.5 + radius * np.cos(angle)
        center_y = 0.5 + radius * np.sin(angle)
        width = rng.uniform(0.05, 0.15)
        weight = rng.uniform(-1.0, 1.0)
        field += weight * np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * width**2))

    field *= mask
    masked_mean = field[mask].mean() if np.any(mask) else 0.0
    field = (field - masked_mean) * mask
    scale = np.max(np.abs(field)) if np.any(mask) else 0.0
    if scale < 1.0e-12:
        return np.zeros_like(field)
    return field / scale


def seeded_disk_profile(
    xx: np.ndarray,
    yy: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    radial = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    base = gaussian_blob_field(
        xx,
        yy,
        mask,
        seed=seed,
        amplitude_range=(0.30, 0.95),
        width_range=(0.04, 0.12),
        num_blobs=4,
    )
    ring = np.exp(-((radial - 0.22) ** 2) / (2.0 * 0.05**2)) * mask
    ring_scale = np.max(ring) if np.any(mask) else 0.0
    if ring_scale > 1.0e-12:
        ring = ring / ring_scale
    profile = 0.7 * base + 0.3 * ring
    scale = np.max(profile) if np.any(mask) else 0.0
    if scale < 1.0e-12:
        return np.zeros_like(profile)
    return (profile / scale) * mask


def solve_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    parameters: np.ndarray,
    mask: np.ndarray,
    config: FisherKPPDiskConfig,
    *,
    history_fraction: float,
    num_snapshots: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    diffusion, growth_rate, initial_amplitude, carrying_capacity_contrast, initial_condition_seed = parameters
    dx = float(x[1] - x[0])
    front_time = 0.95 / np.sqrt(max(diffusion * growth_rate, 1.0e-12))
    reaction_time = 5.0 / growth_rate
    steady_time_estimate = max(front_time, reaction_time)
    total_time = history_fraction * steady_time_estimate
    max_dt = 0.18 * dx * dx / diffusion
    times, save_indices, dt, num_steps = stable_time_grid(
        total_time=total_time,
        max_dt=max_dt,
        num_snapshots=num_snapshots,
    )

    xx, yy = np.meshgrid(x, y, indexing="xy")
    seed_base = int(round(initial_condition_seed))
    state = float(initial_amplitude) * seeded_disk_profile(
        xx,
        yy,
        mask,
        seed=seed_base,
    )
    state *= mask
    habitat = signed_disk_field(
        xx,
        yy,
        mask,
        seed=deterministic_seed("disk_habitat", seed_base),
    )
    carrying_capacity = np.clip(1.0 + carrying_capacity_contrast * habitat, 0.65, 1.35) * mask

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
        next_state = carrying_capacity * diffused * growth_factor / (
            carrying_capacity + diffused * (growth_factor - 1.0) + 1.0e-12
        )
        next_state = np.clip(next_state, 0.0, 1.55)
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
        if trajectory_index < config.train_trajectories:
            states, times, steady_time = solve_trajectory(
                x,
                y,
                parameters,
                mask,
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
                mask,
                config,
                history_fraction=config.test_history_fraction,
                num_snapshots=config.test_num_snapshots,
            )
            assert states.shape[0] == config.test_num_snapshots
            test_states.append(states)
            test_times.append(times)
        steady_time_estimates[trajectory_index] = steady_time

    metadata = {
        "case": "fisher_kpp_disk",
        "equation": "u_t = diffusion * Laplacian(u) + growth_rate * u * (1 - u / K(x, y))",
        "boundary_conditions": "Embedded disk with homogeneous Dirichlet boundary values",
        "notes": "The disk case now uses richer multi-seed initial colonies and a deterministic heterogeneous carrying-capacity field, which produces interacting invasion fronts before the held-out test approaches a spatially varying steady state.",
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
    parser.add_argument("--train-num-snapshots", type=int, default=FisherKPPDiskConfig.train_num_snapshots)
    parser.add_argument("--test-num-snapshots", type=int, default=FisherKPPDiskConfig.test_num_snapshots)
    parser.add_argument("--train-trajectories", type=int, default=FisherKPPDiskConfig.train_trajectories)
    parser.add_argument("--total-trajectories", type=int, default=FisherKPPDiskConfig.total_trajectories)
    parser.add_argument("--train-history-fraction", type=float, default=FisherKPPDiskConfig.train_history_fraction)
    parser.add_argument("--test-history-fraction", type=float, default=FisherKPPDiskConfig.test_history_fraction)
    args = parser.parse_args()
    return FisherKPPDiskConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        grid_size=args.grid_size,
        train_num_snapshots=args.train_num_snapshots,
        test_num_snapshots=args.test_num_snapshots,
        train_trajectories=args.train_trajectories,
        total_trajectories=args.total_trajectories,
        train_history_fraction=args.train_history_fraction,
        test_history_fraction=args.test_history_fraction,
    )


def main() -> None:
    generate_dataset(parse_args())


if __name__ == "__main__":
    main()
