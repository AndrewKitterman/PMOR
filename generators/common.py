from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

Array = np.ndarray
DEFAULT_TRAIN_TRAJECTORIES = 50
DEFAULT_TOTAL_TRAJECTORIES = 51


def parameter_table(
    bounds: list[tuple[str, float, float]],
    *,
    total_trajectories: int,
    seed: int,
) -> tuple[Array, list[str]]:
    """Build a deterministic Latin hypercube sample over box constraints."""
    if total_trajectories < 2:
        raise ValueError("Expected at least two trajectories so one can be held out for testing.")

    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    sample = sampler.random(total_trajectories)
    lower = np.array([item[1] for item in bounds], dtype=np.float64)
    upper = np.array([item[2] for item in bounds], dtype=np.float64)
    scaled = qmc.scale(sample, lower, upper)
    names = [item[0] for item in bounds]
    return scaled.astype(np.float64), names


def stable_time_grid(
    *,
    total_time: float,
    max_dt: float,
    num_snapshots: int,
    oversample: int = 8,
) -> tuple[Array, Array, float, int]:
    """Choose a stable time grid and evenly spaced snapshot indices."""
    if total_time <= 0.0:
        raise ValueError("total_time must be positive")
    if max_dt <= 0.0:
        raise ValueError("max_dt must be positive")
    if num_snapshots < 2:
        raise ValueError("num_snapshots must be at least 2")

    min_steps = max(num_snapshots - 1, oversample * (num_snapshots - 1))
    num_steps = max(min_steps, int(np.ceil(total_time / max_dt)))
    dt = total_time / num_steps
    save_indices = np.linspace(0, num_steps, num_snapshots, dtype=np.int64)
    times = save_indices.astype(np.float64) * dt
    return times, save_indices, dt, num_steps


def sine_series(x: Array, *, seed: int, amplitude: float, modes: int = 4) -> Array:
    """Smooth 1D field with homogeneous Dirichlet endpoints."""
    rng = np.random.default_rng(seed)
    field = np.zeros_like(x, dtype=np.float64)
    for mode in range(1, modes + 1):
        coeff = rng.uniform(-1.0, 1.0)
        field += coeff * np.sin(mode * np.pi * x)

    norm = np.max(np.abs(field))
    if norm < 1.0e-12:
        return np.zeros_like(field)
    return amplitude * field / norm


def periodic_series(x: Array, *, seed: int, amplitude: float, modes: int = 4) -> Array:
    """Smooth periodic 1D field with zero spatial mean."""
    rng = np.random.default_rng(seed)
    field = np.zeros_like(x, dtype=np.float64)
    for mode in range(1, modes + 1):
        sin_coeff = rng.uniform(-1.0, 1.0)
        cos_coeff = rng.uniform(-1.0, 1.0)
        phase = 2.0 * np.pi * mode * x
        field += sin_coeff * np.sin(phase) + cos_coeff * np.cos(phase)

    field -= field.mean()
    norm = np.max(np.abs(field))
    if norm < 1.0e-12:
        return np.zeros_like(field)
    return amplitude * field / norm


def disk_mask(
    xx: Array,
    yy: Array,
    *,
    center: tuple[float, float] = (0.5, 0.5),
    radius: float = 0.45,
) -> Array:
    """Boolean disk mask embedded in a Cartesian grid."""
    cx, cy = center
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def gaussian_blob_field(
    xx: Array,
    yy: Array,
    mask: Array,
    *,
    seed: int,
    amplitude_range: tuple[float, float] = (0.25, 0.85),
    width_range: tuple[float, float] = (0.05, 0.16),
    num_blobs: int = 3,
) -> Array:
    """Positive smooth 2D field built from deterministic Gaussian blobs."""
    rng = np.random.default_rng(seed)
    field = np.zeros_like(xx, dtype=np.float64)
    for _ in range(num_blobs):
        radius = np.sqrt(rng.uniform(0.0, 1.0)) * 0.28
        angle = rng.uniform(0.0, 2.0 * np.pi)
        cx = 0.5 + radius * np.cos(angle)
        cy = 0.5 + radius * np.sin(angle)
        width = rng.uniform(*width_range)
        amplitude = rng.uniform(*amplitude_range)
        field += amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * width**2))

    field *= mask
    max_value = float(np.max(field)) if field.size else 0.0
    if max_value > 0.0:
        field /= max_value
    return field


def deterministic_seed(*parts: Any) -> int:
    """Derive a reproducible 32-bit seed from arbitrary hashable parts."""
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def as_metadata(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, default=str)


def assert_finite(name: str, array: Array) -> None:
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name} contains non-finite values")


def flatten_trajectories(state_sequences: list[Array], time_sequences: list[Array]) -> dict[str, Array]:
    """Flatten per-trajectory states into sequential snapshot arrays."""
    lengths = np.array([states.shape[0] for states in state_sequences], dtype=np.int32)
    if np.any(lengths <= 0):
        raise ValueError("Every trajectory must contain at least one snapshot.")

    offsets = np.zeros(lengths.size + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths, dtype=np.int32)
    state_array = np.concatenate([states.astype(np.float32, copy=False) for states in state_sequences], axis=0)
    time_array = np.concatenate([times.astype(np.float32, copy=False) for times in time_sequences], axis=0)
    trajectory_ids = np.concatenate(
        [np.full(length, idx, dtype=np.int32) for idx, length in enumerate(lengths)],
        axis=0,
    )
    return {
        "states": state_array,
        "times": time_array,
        "trajectory_ids": trajectory_ids,
        "trajectory_lengths": lengths,
        "trajectory_offsets": offsets,
    }


def save_dataset(
    output_path: Path,
    *,
    case_name: str,
    split_name: str,
    state_sequences: list[Array],
    time_sequences: list[Array],
    parameter_matrix: Array,
    parameter_names: list[str],
    steady_time_estimates: Array,
    grid_data: dict[str, Array],
    metadata: dict[str, Any],
) -> None:
    """Persist a sequential-snapshot dataset in a compact NPZ file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flattened = flatten_trajectories(state_sequences, time_sequences)

    payload: dict[str, Any] = {
        "case_name": np.array(case_name),
        "split_name": np.array(split_name),
        "parameter_names": np.array(parameter_names, dtype="U64"),
        "parameter_matrix": np.asarray(parameter_matrix, dtype=np.float64),
        "steady_time_estimates": np.asarray(steady_time_estimates, dtype=np.float64),
        "metadata_json": np.array(as_metadata(metadata)),
    }
    payload.update(flattened)
    for key, value in grid_data.items():
        payload[key] = np.asarray(value)

    np.savez_compressed(output_path, **payload)
