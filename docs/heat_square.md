# Heat Equation on the Unit Square

## Equation

The generator solves

`u_t = kappa * (u_xx + u_yy) + q(x, y, t)`

on the unit square with parameterized Dirichlet values on the left, right, bottom, and top boundaries. The source term is deterministic and exponentially decays to zero, so the steady limit remains the boundary-value solution.

## Parameters

- `diffusivity` in `[0.08, 0.24]`
- `left_boundary` in `[-0.65, 0.65]`
- `right_boundary` in `[-0.65, 0.65]`
- `bottom_boundary` in `[-0.65, 0.65]`
- `top_boundary` in `[-0.65, 0.65]`
- `initial_amplitude` in `[0.08, 0.35]`
- `source_amplitude` in `[0.00, 0.16]`
- `initial_condition_seed` derived deterministically per trajectory

## Discretization

- Grid: `48 x 48`
- Time integration: explicit five-point stencil with a conservative diffusion stability limit
- Initial conditions: deterministic smooth modes plus signed off-center thermal blobs
- Source decay: fixed at `exp(-1.4 t)`
- Training history: `24` snapshots over the first `30%` of a diffusion-based steady-state estimate
- Test history: `64` snapshots over `200%` of the same estimate so the final snapshots sit near the steady boundary-value solution

This is still the clean boundary-condition-parameterized 2D case in the repo, but with a visibly richer transient driven by interior heating and cooling patterns that fade away over time.

## Files

- `data/heat_square/train.npz`
- `data/heat_square/test.npz`
