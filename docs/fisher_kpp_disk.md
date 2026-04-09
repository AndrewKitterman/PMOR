# Fisher-KPP on a Disk

## Equation

The generator solves

`u_t = D * Laplacian(u) + r * u * (1 - u)`

on a disk embedded in a Cartesian square grid. Values outside the mask are fixed to zero, which gives a simple homogeneous Dirichlet boundary treatment on the disk.

## Parameters

- `diffusion` in `[0.010, 0.040]`
- `growth_rate` in `[0.70, 1.60]`
- `initial_amplitude` in `[0.35, 0.95]`
- `initial_condition_seed` derived deterministically per trajectory

## Discretization

- Grid: `48 x 48` Cartesian grid with a disk mask of radius `0.45`
- Time integration: explicit diffusion step plus exact logistic reaction update
- Saved history: `24` snapshots over the first `25%` of a combined reaction-diffusion relaxation estimate

This is the nonlinear 2D geometry-variation case in the repo.

## Files

- `data/fisher_kpp_disk/train.npz`
- `data/fisher_kpp_disk/test.npz`
