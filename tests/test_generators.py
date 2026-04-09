from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generators.generate_burgers_1d import (
    Burgers1DConfig,
    build_parameter_matrix as build_burgers_parameter_matrix,
    generate_dataset as generate_burgers_dataset,
)
from generators.generate_damped_wave_1d import (
    DampedWave1DConfig,
    build_parameter_matrix as build_damped_wave_parameter_matrix,
    generate_dataset as generate_damped_wave_dataset,
)
from generators.generate_fisher_kpp_disk import (
    FisherKPPDiskConfig,
    build_parameter_matrix as build_fisher_kpp_parameter_matrix,
    generate_dataset as generate_fisher_kpp_dataset,
)
from generators.generate_heat_square import (
    HeatSquareConfig,
    build_parameter_matrix as build_heat_square_parameter_matrix,
    generate_dataset as generate_heat_square_dataset,
)


def verify_dataset(
    dataset: np.lib.npyio.NpzFile,
    *,
    expected_case_name: str,
    expected_trajectories: int,
    expected_snapshots: int,
    expected_state_shape: tuple[int, ...],
) -> None:
    total_snapshots = expected_trajectories * expected_snapshots
    states = dataset["states"]
    times = dataset["times"]
    trajectory_ids = dataset["trajectory_ids"]
    trajectory_lengths = dataset["trajectory_lengths"]
    trajectory_offsets = dataset["trajectory_offsets"]
    parameter_matrix = dataset["parameter_matrix"]
    parameter_names = dataset["parameter_names"]
    steady_time_estimates = dataset["steady_time_estimates"]

    assert states.shape == (total_snapshots, *expected_state_shape)
    assert np.all(np.isfinite(states))
    assert times.shape == (total_snapshots,)
    assert trajectory_ids.shape == (total_snapshots,)
    assert trajectory_lengths.shape == (expected_trajectories,)
    assert trajectory_offsets.shape == (expected_trajectories + 1,)
    assert int(trajectory_offsets[-1]) == total_snapshots
    assert parameter_matrix.shape[0] == expected_trajectories
    assert steady_time_estimates.shape == (expected_trajectories,)
    assert parameter_names[-1] == "initial_condition_seed"
    metadata = json.loads(str(dataset["metadata_json"]))
    assert metadata["case"] == expected_case_name
    assert int(metadata["config"]["num_snapshots"]) == expected_snapshots
    assert np.all(steady_time_estimates > 0.0)

    for trajectory_index in range(expected_trajectories):
        start = int(trajectory_offsets[trajectory_index])
        stop = int(trajectory_offsets[trajectory_index + 1])
        assert stop - start == expected_snapshots
        assert np.all(trajectory_ids[start:stop] == trajectory_index)
        assert abs(float(times[start])) < 1.0e-7
        assert np.all(np.diff(times[start:stop]) >= 0.0)


class GeneratorSmokeTests(unittest.TestCase):
    def test_all_generators_emit_finite_split_datasets(self) -> None:
        cases = [
            (
                "damped_wave_1d",
                lambda root: (
                    DampedWave1DConfig(
                        output_dir=root / "damped_wave_1d",
                        num_points=65,
                        num_snapshots=8,
                        train_trajectories=2,
                        total_trajectories=3,
                    ),
                    generate_damped_wave_dataset,
                    (65,),
                ),
            ),
            (
                "burgers_1d",
                lambda root: (
                    Burgers1DConfig(
                        output_dir=root / "burgers_1d",
                        num_points=96,
                        num_snapshots=8,
                        train_trajectories=2,
                        total_trajectories=3,
                    ),
                    generate_burgers_dataset,
                    (96,),
                ),
            ),
            (
                "heat_square",
                lambda root: (
                    HeatSquareConfig(
                        output_dir=root / "heat_square",
                        grid_size=24,
                        num_snapshots=6,
                        train_trajectories=2,
                        total_trajectories=3,
                    ),
                    generate_heat_square_dataset,
                    (24, 24),
                ),
            ),
            (
                "fisher_kpp_disk",
                lambda root: (
                    FisherKPPDiskConfig(
                        output_dir=root / "fisher_kpp_disk",
                        grid_size=24,
                        num_snapshots=6,
                        train_trajectories=2,
                        total_trajectories=3,
                    ),
                    generate_fisher_kpp_dataset,
                    (24, 24),
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for case_name, builder in cases:
                config, generator, state_shape = builder(root)
                with self.subTest(case=case_name):
                    outputs = generator(config)
                    with np.load(outputs["train"], allow_pickle=False) as train_dataset:
                        verify_dataset(
                            train_dataset,
                            expected_case_name=case_name,
                            expected_trajectories=config.train_trajectories,
                            expected_snapshots=config.num_snapshots,
                            expected_state_shape=state_shape,
                        )
                    with np.load(outputs["test"], allow_pickle=False) as test_dataset:
                        verify_dataset(
                            test_dataset,
                            expected_case_name=case_name,
                            expected_trajectories=config.total_trajectories - config.train_trajectories,
                            expected_snapshots=config.num_snapshots,
                            expected_state_shape=state_shape,
                        )

                        if case_name == "fisher_kpp_disk":
                            mask = test_dataset["mask"].astype(bool)
                            states = test_dataset["states"]
                            self.assertTrue(np.allclose(states[:, ~mask], 0.0))

    def test_committed_default_datasets_match_generator_defaults(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        committed_cases = [
            (
                "damped_wave_1d",
                DampedWave1DConfig(),
                build_damped_wave_parameter_matrix,
                (129,),
            ),
            (
                "burgers_1d",
                Burgers1DConfig(),
                build_burgers_parameter_matrix,
                (192,),
            ),
            (
                "heat_square",
                HeatSquareConfig(),
                build_heat_square_parameter_matrix,
                (48, 48),
            ),
            (
                "fisher_kpp_disk",
                FisherKPPDiskConfig(),
                build_fisher_kpp_parameter_matrix,
                (48, 48),
            ),
        ]

        for case_name, config, parameter_builder, state_shape in committed_cases:
            train_path = repo_root / "data" / case_name / "train.npz"
            test_path = repo_root / "data" / case_name / "test.npz"
            self.assertTrue(train_path.exists(), train_path.as_posix())
            self.assertTrue(test_path.exists(), test_path.as_posix())

            expected_parameters, expected_names = parameter_builder(config)
            with self.subTest(case=case_name):
                with np.load(train_path, allow_pickle=False) as train_dataset:
                    verify_dataset(
                        train_dataset,
                        expected_case_name=case_name,
                        expected_trajectories=config.train_trajectories,
                        expected_snapshots=config.num_snapshots,
                        expected_state_shape=state_shape,
                    )
                    np.testing.assert_array_equal(train_dataset["parameter_names"], np.array(expected_names))
                    np.testing.assert_array_equal(
                        train_dataset["parameter_matrix"],
                        expected_parameters[: config.train_trajectories],
                    )

                with np.load(test_path, allow_pickle=False) as test_dataset:
                    verify_dataset(
                        test_dataset,
                        expected_case_name=case_name,
                        expected_trajectories=config.total_trajectories - config.train_trajectories,
                        expected_snapshots=config.num_snapshots,
                        expected_state_shape=state_shape,
                    )
                    np.testing.assert_array_equal(test_dataset["parameter_names"], np.array(expected_names))
                    np.testing.assert_array_equal(
                        test_dataset["parameter_matrix"],
                        expected_parameters[config.train_trajectories :],
                    )


if __name__ == "__main__":
    unittest.main()
