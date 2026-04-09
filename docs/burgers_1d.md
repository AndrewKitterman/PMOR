# Viscous Burgers in 1D

## Equation

The generator solves

`u_t + (0.5 * u^2)_x = nu * u_xx + f(x, t)`

on a periodic unit interval. The forcing is deterministic, zero mean, and exponentially decaying in time, so the long-time state still relaxes to a constant profile with the same spatial mean.

## Parameters

- `viscosity` in `[0.016, 0.065]`
- `mean_level` in `[-0.45, 0.45]`
- `initial_amplitude` in `[0.22, 0.95]`
- `forcing_amplitude` in `[0.00, 0.18]`
- `initial_condition_seed` derived deterministically per trajectory

## Discretization

- Spatial grid: `192` periodic points
- Flux: local Lax-Friedrichs / Rusanov numerical flux
- Time integration: SSPRK3
- Initial conditions: deterministic mixtures of Fourier modes and wrapped Gaussian bumps for sharper fronts
- Forcing decay: fixed at `exp(-1.25 t)`
- Training history: `32` snapshots over the first `30%` of a viscosity-based decay estimate
- Test history: `64` snapshots over `200%` of the same estimate so the final snapshots sit close to the constant steady state

The mean level is conserved because the forcing is zero mean, so the long-time solution still tends toward a constant profile with that same mean after a more dynamic transient.

## Files

- `data/burgers_1d/train.npz`
- `data/burgers_1d/test.npz`
