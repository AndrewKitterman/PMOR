# Viscous Burgers in 1D

## Equation

The generator solves

`u_t + (0.5 * u^2)_x = nu * u_xx`

on a periodic unit interval.

## Parameters

- `viscosity` in `[0.025, 0.080]`
- `mean_level` in `[-0.35, 0.35]`
- `initial_amplitude` in `[0.18, 0.75]`
- `initial_condition_seed` derived deterministically per trajectory

## Discretization

- Spatial grid: `192` periodic points
- Flux: local Lax-Friedrichs / Rusanov numerical flux
- Time integration: SSPRK3
- Training history: `32` snapshots over the first `25%` of a viscosity-based decay estimate
- Test history: `64` snapshots over `200%` of the same estimate so the final snapshots sit close to the constant steady state

The mean level is conserved, so the long-time solution tends toward a constant profile with that same mean.

## Files

- `data/burgers_1d/train.npz`
- `data/burgers_1d/test.npz`
