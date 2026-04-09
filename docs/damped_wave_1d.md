# Damped Wave in 1D

## Equation

The generator solves

`u_tt + gamma * u_t = c^2 * u_xx`

on `x in [0, 1]` with constant Dirichlet boundary values. The boundary values, wave speed, and damping coefficient are all parameterized.

## Parameters

- `wave_speed` in `[0.8, 1.8]`
- `damping` in `[0.6, 1.4]`
- `left_boundary` in `[-0.25, 0.25]`
- `right_boundary` in `[-0.25, 0.25]`
- `displacement_amplitude` in `[0.06, 0.32]`
- `velocity_amplitude` in `[0.04, 0.20]`
- `initial_condition_seed` derived deterministically per trajectory

## Discretization

- Spatial grid: `129` points on `[0, 1]`
- Time integration: centered explicit damped-wave update with a CFL-based `dt`
- Saved history: `32` snapshots over the first `25%` of a damping-based steady-state estimate

The boundary values stay fixed for all time, and the damped dynamics drive the transient toward the corresponding static profile.

## Files

- `data/damped_wave_1d/train.npz`
- `data/damped_wave_1d/test.npz`
