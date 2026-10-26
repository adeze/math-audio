# Unreleased

- Borrow the existing parameter vector for linear-penalty matrix multiplication instead of cloning it on every scalar objective evaluation. Preserve exact penalty results and input contents; no solver defaults change.

# 0.5.15

## New features

- Added `cobra`, a SACOBRA-style surrogate-assisted constrained optimiser
  (RBF models per objective/constraint, distance-requirement cycle,
  self-adjusting output transform, stagnation kicks, COBYLA polish).
- Added `constrained_bayesian_optimization`, a single-trust-region
  constrained BO arm with constrained EI.
- Added `RbfSurrogate` with cubic/Gaussian/multiquadric kernels and
  holdout model selection (`fit_auto`).
- CMA-ES gained native inequality constraints with adaptive-penalty merit,
  IPOP restarts, and a diagonal-covariance mode.
- Added the `benchmark-constrained` suite (G06, G08, G24, G04, G09,
  rosenbrock-disk) with budget-normalised rungs; COBRA solves 5/6 problems
  against 2/6 for the CMA-ES baseline.
