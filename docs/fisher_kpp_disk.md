# Fisher-KPP on a Disk

## Equation

The generator solves

`u_t = D * Laplacian(u) + r * u * (1 - u / K(x, y))`

on a disk embedded in a Cartesian square grid. Values outside the mask are fixed to zero, which gives a simple homogeneous Dirichlet boundary treatment on the disk. The carrying-capacity field `K(x, y)` is deterministic, spatially varying, and parameterized through its contrast.

## Parameters

- `diffusion` in `[0.008, 0.028]`
- `growth_rate` in `[0.80, 1.90]`
- `initial_amplitude` in `[0.10, 0.75]`
- `carrying_capacity_contrast` in `[0.00, 0.40]`
- `initial_condition_seed` derived deterministically per trajectory

## Discretization

- Grid: `48 x 48` Cartesian grid with a disk mask of radius `0.45`
- Time integration: explicit diffusion step plus exact logistic reaction update
- Initial conditions: deterministic multi-blob colonies blended with a ring-like seed to create interacting invasion fronts
- Training history: `24` snapshots over the first `30%` of a combined front-propagation and reaction relaxation estimate
- Test history: `64` snapshots over `200%` of the same estimate so the final snapshots are much closer to the nonlinear steady state on the disk

This is the nonlinear 2D geometry-variation case in the repo, now with more pronounced front interaction and a spatially varying steady state.

## Files

- `data/fisher_kpp_disk/train.npz`
- `data/fisher_kpp_disk/test.npz`
