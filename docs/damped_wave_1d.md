# Damped Wave in 1D

## Equation

The generator solves

`u_tt + gamma * u_t = c^2 * u_xx + f(x)`

on `x in [0, 1]` with constant Dirichlet boundary values. The boundary values, wave speed, damping coefficient, and forcing amplitude are parameterized. The forcing shape itself is deterministic and derived from the trajectory seed.

## Parameters

- `wave_speed` in `[0.85, 2.10]`
- `damping` in `[0.30, 1.10]`
- `left_boundary` in `[-0.45, 0.45]`
- `right_boundary` in `[-0.45, 0.45]`
- `displacement_amplitude` in `[0.10, 0.45]`
- `velocity_amplitude` in `[0.08, 0.30]`
- `forcing_amplitude` in `[0.00, 0.65]`
- `initial_condition_seed` derived deterministically per trajectory

## Discretization

- Spatial grid: `129` points on `[0, 1]`
- Time integration: centered explicit damped-wave update with a conservative CFL-based `dt`
- Initial conditions: deterministic mixtures of modal content and localized bumps in both displacement and velocity
- Training history: `32` snapshots over the first `30%` of a damping-based steady-state estimate
- Test history: `64` snapshots over `200%` of the same estimate so the final snapshots lie near the forced static profile

The boundary values stay fixed for all time, and the damped dynamics drive the transient toward the corresponding forced static profile after a more visibly oscillatory ringing phase.

## Files

- `data/damped_wave_1d/train.npz`
- `data/damped_wave_1d/test.npz`
