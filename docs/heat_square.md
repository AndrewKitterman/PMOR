# Heat Equation on the Unit Square

## Equation

The generator solves

`u_t = kappa * (u_xx + u_yy)`

on the unit square with parameterized Dirichlet values on the left, right, bottom, and top boundaries.

## Parameters

- `diffusivity` in `[0.08, 0.28]`
- `left_boundary` in `[-0.40, 0.40]`
- `right_boundary` in `[-0.40, 0.40]`
- `bottom_boundary` in `[-0.40, 0.40]`
- `top_boundary` in `[-0.40, 0.40]`
- `initial_amplitude` in `[0.05, 0.24]`
- `initial_condition_seed` derived deterministically per trajectory

## Discretization

- Grid: `48 x 48`
- Time integration: explicit five-point stencil with a conservative diffusion stability limit
- Training history: `24` snapshots over the first `25%` of a diffusion-based steady-state estimate
- Test history: `64` snapshots over `200%` of the same estimate so the final snapshots sit near the steady boundary-value solution

This is the clean boundary-condition-parameterized 2D case in the repo.

## Files

- `data/heat_square/train.npz`
- `data/heat_square/test.npz`
