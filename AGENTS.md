# Agent Notes

This repository is set up so automated contributors can add or revise PDE generators without breaking the dataset contract.

## Non-Negotiable Conventions

- Keep the default split at `50` training trajectories and `1` held-out test trajectory.
- Save snapshots sequentially by trajectory in a single flattened `states` array.
- Store enough metadata to reconstruct each trajectory from `trajectory_offsets` and `parameter_matrix`.
- Use deterministic parameter sampling and deterministic initial-condition seeds.
- Reject or fail fast on non-finite states.
- Prefer numerically conservative schemes over aggressive ones; this repo values reliable finite data more than flashy solvers.

## Current Case Roster

- `generators/generate_damped_wave_1d.py`
- `generators/generate_burgers_1d.py`
- `generators/generate_heat_square.py`
- `generators/generate_fisher_kpp_disk.py`

## When Editing

- Update the matching file in `docs/` whenever a case changes.
- Run `python -m unittest discover -s tests -v` before committing.
- If you regenerate datasets, keep the train/test file names stable so downstream notebooks do not drift.
- Avoid adding nondeterministic dependencies unless they are clearly justified and tested on the active Python version.
